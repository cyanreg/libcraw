/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The device, the bridge and the display.
 *
 * libplacebo creates the one VkDevice this program uses, with libcraw's
 * feature requirements chained into it; libcraw is handed the raw handles.
 * A decoded frame never leaves the GPU: FFmpeg's AVVkFrame image is read by
 * libcraw's compute pipeline straight into a libplacebo texture, which the
 * renderer then tone maps to the swapchain. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/mem.h>
#include <libavutil/time.h>

#include <libplacebo/gpu.h>
#include <libplacebo/swapchain.h>
#include <libplacebo/tone_mapping.h>
#include <libplacebo/gamut_mapping.h>
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>
/* Header-only in C, and it needs the Vulkan header first for AVVkFrame
 * support; view.h has already included it. */
#include <libplacebo/utils/libav.h>

#include "view.h"

/* BT.2408 diffuse white; a pipeline value of 1.0 is diffuse white, so the
 * sensor clip lands at this times the profile gain. */
#define VIEW_REF_WHITE_NITS 203.0f

/* --------------------------------------------------------------- tables */

static const struct pl_tone_map_function *const view_tone_map_fns[] = {
    &pl_tone_map_bt2390,
    &pl_tone_map_spline,
    &pl_tone_map_reinhard,
    &pl_tone_map_clip,
};

const char *const view_tone_map_names[] = {
    "bt.2390", "spline", "reinhard", "clip",
};

const int view_nb_tone_maps = (int)(sizeof(view_tone_map_fns) / sizeof(view_tone_map_fns[0]));

static const struct pl_gamut_map_function *const view_gamut_map_fns[] = {
    &pl_gamut_map_perceptual,
    &pl_gamut_map_softclip,
    &pl_gamut_map_relative,
    &pl_gamut_map_clip,
    &pl_gamut_map_desaturate,
};

/* One word each: the same strings name the choice in the panel and on the
 * command line. */
const char *const view_gamut_map_names[] = {
    "perceptual", "softclip", "relative", "clip", "desaturate",
};

const int view_nb_gamut_maps = (int)(sizeof(view_gamut_map_fns) / sizeof(view_gamut_map_fns[0]));

/* What a .cube LUT expects at its input. Anything but the first entry makes
 * the LUT a PL_LUT_CONVERSION from that space to a BT.709 display, which is
 * how a camera or creative cube is meant to be applied. Supporting another
 * authoring space is one row here. */
static const struct pl_color_space view_lut_space_in[] = {
    { 0 },                                                          /* native */
    { .primaries = PL_COLOR_PRIM_BT_709,  .transfer = PL_COLOR_TRC_BT_1886 },
    { .primaries = PL_COLOR_PRIM_BT_2020, .transfer = PL_COLOR_TRC_HLG    },
    { .primaries = PL_COLOR_PRIM_BT_2020, .transfer = PL_COLOR_TRC_LINEAR },
};

/* One word each: the same strings name the choice in the panel and on the
 * command line. */
const char *const view_lut_space_names[] = {
    "native", "bt709", "hlg2020", "linear2020",
};

const int view_nb_lut_spaces = (int)CRAW_ARRAY_ELEMS(view_lut_space_in);

void view_lut_apply_space(struct pl_custom_lut *lut, int space)
{
    if (!lut)
        return;

    space = CRAW_CLAMP(space, 0, view_nb_lut_spaces - 1);

    if (!space) {
        lut->color_in = (struct pl_color_space) { 0 };
        lut->color_out = (struct pl_color_space) { 0 };
        lut->repr_in = (struct pl_color_repr) { 0 };
        lut->repr_out = (struct pl_color_repr) { 0 };
        return;
    }

    lut->color_in = view_lut_space_in[space];
    lut->color_out = (struct pl_color_space) {
        .primaries = PL_COLOR_PRIM_BT_709,
        .transfer  = PL_COLOR_TRC_BT_1886,
    };
    lut->repr_in = lut->repr_out = (struct pl_color_repr) {
        .sys    = PL_COLOR_SYSTEM_RGB,
        .levels = PL_COLOR_LEVELS_PC,
    };
}

const struct pl_tone_map_function *view_tone_map(int idx)
{
    return view_tone_map_fns[CRAW_CLAMP(idx, 0, view_nb_tone_maps - 1)];
}

const struct pl_gamut_map_function *view_gamut_map(int idx)
{
    return view_gamut_map_fns[CRAW_CLAMP(idx, 0, view_nb_gamut_maps - 1)];
}

/* --------------------------------------------------------------- device */

static void view_pl_log(void *priv, enum pl_log_level level, const char *msg)
{
    (void)priv;
    fprintf(stderr, "[libplacebo %d] %s\n", (int)level, msg);
}

/* libcraw states its requirements as feature structs with the bits it needs
 * set to VK_TRUE. Every one of them is a run of VkBool32 after the common
 * header, so the check is the same loop for all of them. */
static bool view_feats_ok(const void *req, const void *sup, size_t bytes, size_t skip)
{
    const VkBool32 *r = (const VkBool32 *)((const char *)req + skip);
    const VkBool32 *s = (const VkBool32 *)((const char *)sup + skip);
    size_t n = (bytes - skip) / sizeof(VkBool32);

    for (size_t i = 0; i < n; i++)
        if (r[i] && !s[i])
            return false;

    return true;
}

#define VIEW_FEATS_OK(req, sup) \
    view_feats_ok(&(req), &(sup), sizeof(req), sizeof(VkBaseOutStructure))

static bool view_device_ok(VkPhysicalDevice pd, const CRAWVulkanRequirements *rq,
                           VkSurfaceKHR surface, char *name, size_t name_size,
                           int *score, bool verbose)
{
    VkPhysicalDeviceVulkan14Features f14 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
    VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = &f14 };
    VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13 };
    VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12 };
    VkPhysicalDeviceFeatures2 f2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11 };
    VkPhysicalDeviceProperties props;
    VkQueueFamilyProperties *qprops = NULL;
    VkExtensionProperties *exts = NULL;
    uint32_t nb_q = 0, nb_ext = 0;
    bool ok = false;
    bool have_gfx = false, have_present = surface == VK_NULL_HANDLE;

    vkGetPhysicalDeviceProperties(pd, &props);
    snprintf(name, name_size, "%s", props.deviceName);

    if (props.apiVersion < rq->min_api_version) {
        if (verbose)
            fprintf(stderr, "  %s: Vulkan %u.%u, libcraw needs %u.%u\n", props.deviceName,
                    VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
                    VK_API_VERSION_MAJOR(rq->min_api_version), VK_API_VERSION_MINOR(rq->min_api_version));
        return false;
    }

    vkGetPhysicalDeviceFeatures2(pd, &f2);

    if (!view_feats_ok(&rq->required_features, &f2.features,
                       sizeof(VkPhysicalDeviceFeatures), 0) ||
        !VIEW_FEATS_OK(rq->required_features_11, f11) ||
        !VIEW_FEATS_OK(rq->required_features_12, f12) ||
        !VIEW_FEATS_OK(rq->required_features_13, f13) ||
        !VIEW_FEATS_OK(rq->required_features_14, f14)) {
        if (verbose)
            fprintf(stderr, "  %s: missing a feature libcraw requires\n", props.deviceName);
        return false;
    }

    vkEnumerateDeviceExtensionProperties(pd, NULL, &nb_ext, NULL);
    if (nb_ext) {
        exts = calloc(nb_ext, sizeof(*exts));
        if (!exts)
            return false;
        vkEnumerateDeviceExtensionProperties(pd, NULL, &nb_ext, exts);
    }

    for (uint32_t i = 0; i < rq->nb_required_device_extensions; i++) {
        bool found = false;
        for (uint32_t j = 0; j < nb_ext; j++)
            found |= !strcmp(exts[j].extensionName, rq->required_device_extensions[i]);
        if (!found) {
            if (verbose)
                fprintf(stderr, "  %s: no %s\n", props.deviceName,
                        rq->required_device_extensions[i]);
            goto done;
        }
    }

    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nb_q, NULL);
    if (nb_q) {
        qprops = calloc(nb_q, sizeof(*qprops));
        if (!qprops)
            goto done;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &nb_q, qprops);
    }

    for (uint32_t i = 0; i < nb_q; i++) {
        const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        VkBool32 present = VK_FALSE;

        if ((qprops[i].queueFlags & need) == need)
            have_gfx = true;
        if (surface != VK_NULL_HANDLE) {
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
            have_present |= present == VK_TRUE;
        }
    }

    if (!have_gfx || !have_present) {
        if (verbose)
            fprintf(stderr, "  %s: no graphics+compute queue family%s\n", props.deviceName,
                    have_present ? "" : " that can present");
        goto done;
    }

    switch (props.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   *score = 4; break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: *score = 3; break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    *score = 2; break;
    default:                                     *score = 1; break;
    }
    ok = true;

done:
    free(qprops);
    free(exts);
    return ok;
}

/* libplacebo would happily pick the "best" device by type, but on this
 * machine that can be an Intel iGPU, which lacks the formatless storage
 * image reads libcraw's shaders need. Choose among the devices that can
 * actually run the pipeline instead, and hand the winner to libplacebo. */
static VkPhysicalDevice view_pick_device(struct view_gpu *g, const struct view_cfg *cfg)
{
    CRAWVulkanRequirements rq;
    VkPhysicalDevice *devs = NULL, best = VK_NULL_HANDLE;
    uint32_t nb = 0;
    int best_score = -1;

    craw_vulkan_get_requirements(&rq);

    vkEnumeratePhysicalDevices(g->inst->instance, &nb, NULL);
    if (!nb) {
        fprintf(stderr, "no Vulkan devices\n");
        return VK_NULL_HANDLE;
    }

    devs = calloc(nb, sizeof(*devs));
    if (!devs)
        return VK_NULL_HANDLE;
    vkEnumeratePhysicalDevices(g->inst->instance, &nb, devs);

    if (cfg->verbose)
        fprintf(stderr, "Vulkan devices:\n");

    for (uint32_t i = 0; i < nb; i++) {
        char name[256] = { 0 };
        int score = 0;

        if (!view_device_ok(devs[i], &rq, g->surface, name, sizeof(name), &score,
                            cfg->verbose))
            continue;
        if (cfg->device && !strstr(name, cfg->device)) {
            if (cfg->verbose)
                fprintf(stderr, "  %s: usable, but does not match `%s`\n", name, cfg->device);
            continue;
        }
        if (cfg->verbose)
            fprintf(stderr, "  %s: usable (score %d)\n", name, score);

        if (score > best_score) {
            best_score = score;
            best = devs[i];
        }
    }

    free(devs);

    if (best == VK_NULL_HANDLE)
        fprintf(stderr, "no Vulkan device can run libcraw's pipeline%s%s\n",
                cfg->device ? " and matches " : "", cfg->device ? cfg->device : "");

    return best;
}

void view_gpu_lock(struct view_gpu *g)
{
    g->vk->lock_queue(g->vk, g->qf, 0);
}

void view_gpu_unlock(struct view_gpu *g)
{
    g->vk->unlock_queue(g->vk, g->qf, 0);
}

static void view_craw_log(void *opaque, enum CRAWLogLevel level, const char *fmt,
                          va_list args)
{
    (void)opaque;
    (void)level;
    vfprintf(stderr, fmt, args);
}

int view_gpu_open(struct view_gpu *g, const struct view_cfg *cfg, void *sdl_window)
{
    SDL_Window *win = sdl_window;
    CRAWVulkanRequirements rq;
    CRAWContextOptions copts = { 0 };
    CRAWVulkanInitParams init = { 0 };
    VkPhysicalDeviceProperties props;
    VkPhysicalDevice pd;
    const char * const *sdl_exts = NULL;
    const char **av_inst_exts = NULL;
    int nb_av_inst_exts = 0;
    Uint32 nb_sdl_exts = 0;
    int ret;

    craw_vulkan_get_requirements(&rq);

    g->log = pl_log_create(PL_API_VER, pl_log_params(
        .log_cb    = view_pl_log,
        .log_level = cfg->verbose ? PL_LOG_DEBUG : PL_LOG_WARN,
    ));
    if (!g->log)
        return CRAW_ERROR(ENOMEM);

    /* SDL3 hands back the list it owns, which lives as long as the video
     * subsystem does, so there is nothing to copy or free here. */
    if (win) {
        sdl_exts = SDL_Vulkan_GetInstanceExtensions(&nb_sdl_exts);
        if (!sdl_exts) {
            fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions: %s\n", SDL_GetError());
            return CRAW_ERROR(ENOTSUP);
        }
    }

    /* Whatever FFmpeg would like on top; it only ever uses what is there. */
    av_inst_exts = av_vk_get_optional_instance_extensions(&nb_av_inst_exts);

    g->inst = pl_vk_inst_create(g->log, pl_vk_inst_params(
        .debug              = cfg->debug,
        .extensions         = sdl_exts,
        .num_extensions     = (int)nb_sdl_exts,
        .opt_extensions     = av_inst_exts,
        .num_opt_extensions = nb_av_inst_exts,
    ));
    av_free(av_inst_exts);
    if (!g->inst) {
        fprintf(stderr, "cannot create a Vulkan instance\n");
        return CRAW_ERROR(ENOTSUP);
    }

    if (win && !SDL_Vulkan_CreateSurface(win, g->inst->instance, NULL, &g->surface)) {
        fprintf(stderr, "SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
        return CRAW_ERROR(ENOTSUP);
    }

    pd = view_pick_device(g, cfg);
    if (pd == VK_NULL_HANDLE)
        return CRAW_ERROR(ENODEV);

    /* libcraw's requirements, chained the way libplacebo wants them:
     * everything supported is enabled, everything else is dropped, so the
     * chain only has to outlive the call. */
    rq.required_features_14.pNext = NULL;
    rq.required_features_13.pNext = &rq.required_features_14;
    rq.required_features_12.pNext = &rq.required_features_13;
    rq.required_features_11.pNext = &rq.required_features_12;

    /* FFmpeg's compute decoders declare SPIR-V capabilities of their own.
     * The prores_raw shaders need Int16 and ExpectAssumeKHR, and neither
     * libcraw nor libplacebo asks for them, so a validated device rejects
     * the shader modules and the hwaccel falls back to software. */
    rq.required_features.shaderInt16 = VK_TRUE;
    rq.required_features_11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    rq.required_features_14.shaderExpectAssume = VK_TRUE;

    VkPhysicalDeviceFeatures2 feats = {
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext    = &rq.required_features_11,
        .features = rq.required_features,
    };

    /* Video decoding is the one part of FFmpeg's optional extension list
     * worth enabling by hand: none of these has a feature bit to go with
     * it, and without them the HEVC hwaccel refuses to start and a 5.9K
     * clip decodes on the CPU. The rest of that list is deliberately left
     * alone, because each of those extensions does have a feature FFmpeg
     * would normally turn on when it creates the device itself, and
     * enabling e.g. VK_EXT_shader_object without `shaderObject` makes the
     * decoder take a path the device then rejects. */
    static const char *const video_exts[] = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
    };
    const char *opt_exts[CRAW_ARRAY_ELEMS(video_exts) + 8];
    int nb_opt_exts = 0;

    for (uint32_t i = 0; i < rq.nb_optional_device_extensions &&
                         nb_opt_exts < (int)CRAW_ARRAY_ELEMS(opt_exts); i++)
        opt_exts[nb_opt_exts++] = rq.optional_device_extensions[i];
    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(video_exts) &&
                       nb_opt_exts < (int)CRAW_ARRAY_ELEMS(opt_exts); i++)
        opt_exts[nb_opt_exts++] = video_exts[i];

    /* One queue family for everything. The decoder, the raw pipeline, the UI
     * and the renderer then share images without a single ownership
     * transfer, which is worth more here than asynchronous compute. */
    g->vk = pl_vulkan_create(g->log, pl_vulkan_params(
        .instance        = g->inst->instance,
        .get_proc_addr   = g->inst->get_proc_addr,
        .surface         = g->surface,
        .device          = pd,
        .features        = &feats,
        .async_transfer  = false,
        .async_compute   = false,
        .queue_count     = 1,
        .extra_queues    = VIEW_EXTRA_QUEUES,
        .extensions      = rq.required_device_extensions,
        .num_extensions  = (int)rq.nb_required_device_extensions,
        .opt_extensions  = opt_exts,
        .num_opt_extensions = nb_opt_exts,
    ));
    if (!g->vk) {
        fprintf(stderr, "cannot create the Vulkan device\n");
        return CRAW_ERROR(ENOTSUP);
    }
    g->gpu = g->vk->gpu;

    g->qf = g->vk->queue_compute.index;
    g->queue_flags = g->vk->queue_compute.flags;
    if (g->qf != g->vk->queue_graphics.index) {
        /* Should not happen with async compute off, but the UI renders with
         * a graphics pipeline and must share the family. */
        fprintf(stderr, "libplacebo split the graphics (%u) and compute (%u) queue "
                        "families; using graphics for everything\n",
                g->vk->queue_graphics.index, g->qf);
        g->qf = g->vk->queue_graphics.index;
        g->queue_flags = g->vk->queue_graphics.flags;
    }

    /* libplacebo may create the queues with
     * VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR, and a queue
     * created with any flag can only be retrieved through
     * vkGetDeviceQueue2 with the same flags. */
    vkGetDeviceQueue2(g->vk->device, &(VkDeviceQueueInfo2) {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
        .flags            = g->queue_flags,
        .queueFamilyIndex = g->qf,
        .queueIndex       = 0,
    }, &g->queue);
    if (!g->queue) {
        fprintf(stderr, "cannot get queue 0 of family %u\n", g->qf);
        return CRAW_ERROR(ENODEV);
    }

    vkGetPhysicalDeviceProperties(g->vk->phys_device, &props);
    g->timestamp_period = props.limits.timestampPeriod;

    if (cfg->verbose)
        fprintf(stderr, "device %s, Vulkan %u.%u, queue family %u\n", props.deviceName,
                VK_API_VERSION_MAJOR(g->vk->api_version),
                VK_API_VERSION_MINOR(g->vk->api_version), g->qf);

    if (win) {
        /* --bench measures what the pipeline can do, which vsync would cap
         * long before the answer is interesting. */
        g->sw = pl_vulkan_create_swapchain(g->vk, pl_vulkan_swapchain_params(
            .surface      = g->surface,
            .present_mode = cfg->bench ? VK_PRESENT_MODE_IMMEDIATE_KHR
                                       : VK_PRESENT_MODE_FIFO_KHR,
        ));
        if (!g->sw) {
            fprintf(stderr, "cannot create the swapchain\n");
            return CRAW_ERROR(ENOTSUP);
        }
    }

    g->rr = pl_renderer_create(g->log, g->gpu);
    g->rr_export = pl_renderer_create(g->log, g->gpu);
    if (!g->rr || !g->rr_export)
        return CRAW_ERROR(ENOMEM);

    /* Our own command buffer: libcraw records into it, we submit it. */
    VkCommandPoolCreateInfo pool_info = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g->qf,
    };
    if (vkCreateCommandPool(g->vk->device, &pool_info, NULL, &g->pool) != VK_SUCCESS)
        return CRAW_ERROR(ENOMEM);

    VkCommandBuffer cmds[VIEW_SLOTS];
    VkCommandBufferAllocateInfo cb_info = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = g->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = VIEW_SLOTS,
    };
    if (vkAllocateCommandBuffers(g->vk->device, &cb_info, cmds) != VK_SUCCESS)
        return CRAW_ERROR(ENOMEM);

    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    for (int i = 0; i < VIEW_SLOTS; i++) {
        g->slots[i].cmd = cmds[i];
        if (vkCreateFence(g->vk->device, &fence_info, NULL,
                          &g->slots[i].fence) != VK_SUCCESS)
            return CRAW_ERROR(ENOMEM);
    }

    VkQueryPoolCreateInfo q_info = {
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2 * VIEW_SLOTS,
    };
    if (vkCreateQueryPool(g->vk->device, &q_info, NULL, &g->queries) != VK_SUCCESS)
        return CRAW_ERROR(ENOMEM);

    g->out_sem = pl_vulkan_sem_create(g->gpu, pl_vulkan_sem_params(
        .type          = VK_SEMAPHORE_TYPE_TIMELINE,
        .initial_value = 0,
    ));
    if (g->out_sem == VK_NULL_HANDLE)
        return CRAW_ERROR(ENOMEM);

    init = (CRAWVulkanInitParams) {
        .instance               = g->vk->instance,
        .phys_device            = g->vk->phys_device,
        .device                 = g->vk->device,
        .get_proc_addr          = g->vk->get_proc_addr,
        .queue_family_index     = g->qf,
        .enabled_dev_extensions = g->vk->extensions,
        .nb_enabled_dev_extensions = (uint32_t)g->vk->num_extensions,
    };

    copts.log_cb = view_craw_log;
    copts.log_level = cfg->verbose ? CRAW_LOG_VERBOSE : CRAW_LOG_WARN;

    ret = craw_context_create(&g->ctx, &init, &copts);
    if (ret < 0) {
        fprintf(stderr, "craw_context_create failed: %d\n", ret);
        return ret;
    }

    return 0;
}

void view_gpu_close(struct view_gpu *g)
{
    if (g->vk)
        vkDeviceWaitIdle(g->vk->device);

    craw_pipeline_free(&g->pipe);
    craw_context_free(&g->ctx);

    if (g->vk) {
        for (int i = 0; i < g->nb_src_views; i++)
            vkDestroyImageView(g->vk->device, g->src_views[i].view, NULL);
        g->nb_src_views = 0;

        view_gpu_ui_drop_old(g);
        for (int i = 0; i < VIEW_SLOTS; i++) {
            vkDestroyImageView(g->vk->device, g->slots[i].view, NULL);
            vkDestroyImageView(g->vk->device, g->ui_views[i], NULL);
            g->slots[i].view = g->ui_views[i] = VK_NULL_HANDLE;
        }
    }

    if (g->gpu) {
        view_gpu_unmap(g);
        for (int i = 0; i < 4; i++)
            pl_tex_destroy(g->gpu, &g->map_tex[i]);
        pl_mpv_user_shader_destroy(&g->exposure_hook);
        for (int i = 0; i < VIEW_SLOTS; i++) {
            pl_tex_destroy(g->gpu, &g->slots[i].tex);
            pl_tex_destroy(g->gpu, &g->ui_tex[i]);
        }
        pl_tex_destroy(g->gpu, &g->export_tex);
        pl_vulkan_sem_destroy(g->gpu, &g->out_sem);
    }

    if (g->vk) {
        vkDestroyQueryPool(g->vk->device, g->queries, NULL);
        for (int i = 0; i < VIEW_SLOTS; i++) {
            vkDestroyFence(g->vk->device, g->slots[i].fence, NULL);
            g->slots[i].fence = VK_NULL_HANDLE;
            g->slots[i].cmd = VK_NULL_HANDLE;
        }
        vkDestroyCommandPool(g->vk->device, g->pool, NULL);
        g->queries = VK_NULL_HANDLE;
        g->pool = VK_NULL_HANDLE;
    }

    pl_renderer_destroy(&g->rr_export);
    pl_renderer_destroy(&g->rr);
    pl_swapchain_destroy(&g->sw);
    pl_vulkan_destroy(&g->vk);

    /* SDL made the surface, so SDL unmakes it: it knows which allocator and
     * which loader entry point it used. Still before pl_vk_inst_destroy(),
     * and the window outlives this call. */
    if (g->surface != VK_NULL_HANDLE && g->inst) {
        SDL_Vulkan_DestroySurface(g->inst->instance, g->surface, NULL);
        g->surface = VK_NULL_HANDLE;
    }

    pl_vk_inst_destroy(&g->inst);
    pl_log_destroy(&g->log);
}

/* ------------------------------------------------------------- the size */

int view_gpu_ensure(struct view_gpu *g, uint32_t w, uint32_t h,
                    enum CRAWDebayer debayer)
{
    CRAWPipelineParams pp;
    VkFormat fmt;
    VkImageUsageFlags usage;
    pl_fmt plfmt;
    int ret;

    if (g->pipe && g->width == w && g->height == h && g->pipe_debayer == debayer)
        return 0;

    if (g->pipe) {
        /* The output textures are only rebuilt on a size change; a new
         * debayer method just needs a new pipeline. */
        vkDeviceWaitIdle(g->vk->device);
        for (int i = 0; i < VIEW_SLOTS; i++)
            g->slots[i].busy = false;
        craw_pipeline_free(&g->pipe);
    }

    if (!g->slots[0].tex || g->width != w || g->height != h) {
        plfmt = pl_find_named_fmt(g->gpu, "rgba16hf");
        if (!plfmt || !(plfmt->caps & PL_FMT_CAP_STORABLE)) {
            fprintf(stderr, "the GPU has no storable RGBA16F format\n");
            return CRAW_ERROR(ENOTSUP);
        }

        for (int i = 0; i < VIEW_SLOTS; i++) {
            struct view_slot *sl = &g->slots[i];

            vkDestroyImageView(g->vk->device, sl->view, NULL);
            sl->view = VK_NULL_HANDLE;
            pl_tex_destroy(g->gpu, &sl->tex);
            sl->busy = false;

            sl->tex = pl_tex_create(g->gpu, pl_tex_params(
                .w          = (int)w,
                .h          = (int)h,
                .format     = plfmt,
                .storable   = true,
                .sampleable = true,
            ));
            if (!sl->tex) {
                fprintf(stderr, "cannot create the %ux%u output texture\n", w, h);
                return CRAW_ERROR(ENOMEM);
            }

            sl->image = pl_vulkan_unwrap(g->gpu, sl->tex, &fmt, &usage);
            if (!sl->image || fmt != VK_FORMAT_R16G16B16A16_SFLOAT) {
                fprintf(stderr, "the output texture is not an "
                                "R16G16B16A16_SFLOAT image\n");
                return CRAW_ERROR(ENOTSUP);
            }

            VkImageViewCreateInfo vinfo = {
                .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image    = sl->image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format   = fmt,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .levelCount = 1,
                    .layerCount = 1,
                },
            };
            if (vkCreateImageView(g->vk->device, &vinfo, NULL, &sl->view) != VK_SUCCESS)
                return CRAW_ERROR(ENOMEM);
        }

        g->slot = g->display_slot = 0;
        g->submitted = 0;
    }

    craw_pipeline_params_init(&pp);
    pp.width      = w;
    pp.height     = h;
    pp.cfa        = CRAW_CFA_RGGB;
    pp.debayer    = debayer;
    pp.out_space  = CRAW_CS_BT2020_LINEAR;
    pp.out_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    /* Always built: the panel can switch to auto white balance at any time
     * and a pipeline rebuild mid-clip would drop a frame. */
    pp.flags      = CRAW_PIPELINE_FLAG_AWB_HISTOGRAM;
    /* Enough copies that the previous frame's histogram is still readable
     * while the current one is in flight, which is what lets the auto white
     * balance run without a per-frame stall. */
    pp.nb_awb_slots = 3;

    ret = craw_pipeline_create(g->ctx, &pp, &g->pipe);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_create(%ux%u) failed: %d\n", w, h, ret);
        return ret;
    }

    g->width = w;
    g->height = h;
    g->pipe_debayer = debayer;

    return 0;
}

/* --------------------------------------------------------- frame bridge */

/* The decoder hands out a small pool of AVVkFrames over and over, so a view
 * per image is created once and kept. If the pool is somehow larger than
 * the cache, everything is dropped and rebuilt rather than leaked. */
static VkImageView view_src_view(struct view_gpu *g, VkImage image, VkFormat fmt)
{
    VkImageView view = VK_NULL_HANDLE;

    for (int i = 0; i < g->nb_src_views; i++)
        if (g->src_views[i].image == image)
            return g->src_views[i].view;

    if (g->nb_src_views == VIEW_MAX_SRC_VIEWS) {
        vkDeviceWaitIdle(g->vk->device);
        for (int i = 0; i < g->nb_src_views; i++)
            vkDestroyImageView(g->vk->device, g->src_views[i].view, NULL);
        g->nb_src_views = 0;
    }

    VkImageViewCreateInfo info = {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = fmt,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(g->vk->device, &info, NULL, &view) != VK_SUCCESS)
        return VK_NULL_HANDLE;

    g->src_views[g->nb_src_views++] = (struct view_src_view) {
        .image = image, .view = view,
    };

    return view;
}

int view_gpu_process(struct view_gpu *g, const AVFrame *frame,
                     const CRAWFrameParams *fp, struct view_timing *tm)
{
    int64_t t0 = av_gettime_relative();
    AVHWFramesContext *fc = (AVHWFramesContext *)frame->hw_frames_ctx->data;
    AVVulkanFramesContext *vkfc = fc->hwctx;
    AVVkFrame *vkf = (AVVkFrame *)frame->data[0];
    const VkFormat *vkfmt = av_vkfmt_from_pixfmt(fc->sw_format);
    struct view_slot *sl = &g->slots[g->slot];
    VkImageView src_view;
    CRAWImage src, dst;
    uint64_t hold;
    uint64_t ts[2] = { 0, 0 };
    int ret;

    if (!vkfmt || vkfmt[0] != VK_FORMAT_R16_UNORM) {
        fprintf(stderr, "the decoder's hardware frames are not a single R16_UNORM "
                        "plane; craw-view needs an RGGB Bayer mosaic\n");
        return CRAW_ERROR(ENOTSUP);
    }

    src_view = view_src_view(g, vkf->img[0], vkfmt[0]);
    if (src_view == VK_NULL_HANDLE)
        return CRAW_ERROR(ENOMEM);

    /* The only fence this waits on belongs to the slot about to be reused,
     * which was submitted VIEW_SLOTS frames ago and has normally long since
     * retired. Frame N is never waited for before N + 1 is recorded. */
    if (sl->busy) {
        int64_t tw = av_gettime_relative();

        if (vkWaitForFences(g->vk->device, 1, &sl->fence, VK_TRUE,
                            UINT64_MAX) != VK_SUCCESS)
            return CRAW_ERROR(EIO);

        tm->fence += (double)(av_gettime_relative() - tw) / 1000.0;
        sl->busy = false;

        if (vkGetQueryPoolResults(g->vk->device, g->queries, (uint32_t)(2 * g->slot),
                                  2, sizeof(ts), ts, sizeof(ts[0]),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            g->last_gpu_ms = (double)(ts[1] - ts[0]) * g->timestamp_period / 1.0e6;
    }

    src = (CRAWImage) {
        .image  = vkf->img[0],
        .view   = src_view,
        .format = vkfmt[0],
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .width  = g->width,
        .height = g->height,
    };
    dst = (CRAWImage) {
        .image  = sl->image,
        .view   = sl->view,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .width  = g->width,
        .height = g->height,
    };

    /* libplacebo hands the output image over and signals `hold`; we signal
     * hold + 1 when libcraw is done and give it straight back. */
    hold = g->out_value + 1;
    if (!pl_vulkan_hold_ex(g->gpu, pl_vulkan_hold_params(
            .tex       = sl->tex,
            .layout    = VK_IMAGE_LAYOUT_GENERAL,
            .qf        = g->qf,
            .semaphore = (pl_vulkan_sem) { g->out_sem, hold },
        ))) {
        fprintf(stderr, "cannot hold the output texture\n");
        return CRAW_ERROR(EBUSY);
    }
    g->out_value = hold;

    vkResetQueryPool(g->vk->device, g->queries, (uint32_t)(2 * g->slot), 2);

    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(sl->cmd, &begin) != VK_SUCCESS) {
        ret = CRAW_ERROR(EIO);
        goto release;
    }

    /* The frame's layout, access and semaphore value are only stable while
     * the frames context lock is held. */
    vkfc->lock_frame(fc, vkf);

    vkCmdWriteTimestamp2(sl->cmd, VK_PIPELINE_STAGE_2_NONE, g->queries,
                         (uint32_t)(2 * g->slot));

    VkImageMemoryBarrier2 to_general = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = vkf->access[0],
        .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .oldLayout     = vkf->layout[0],
        .newLayout     = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image         = vkf->img[0],
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };

    if (vkf->queue_family[0] != VK_QUEUE_FAMILY_IGNORED &&
        vkf->queue_family[0] != g->qf) {
        to_general.srcQueueFamilyIndex = vkf->queue_family[0];
        to_general.dstQueueFamilyIndex = g->qf;
    }

    vkCmdPipelineBarrier2(sl->cmd, &(VkDependencyInfo) {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &to_general,
    });

    ret = craw_pipeline_record(g->pipe, sl->cmd, &src, &dst, fp);
    if (ret < 0) {
        vkfc->unlock_frame(fc, vkf);
        vkEndCommandBuffer(sl->cmd);
        goto release;
    }

    vkCmdWriteTimestamp2(sl->cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g->queries,
                         (uint32_t)(2 * g->slot + 1));

    if (vkEndCommandBuffer(sl->cmd) != VK_SUCCESS) {
        vkfc->unlock_frame(fc, vkf);
        ret = CRAW_ERROR(EIO);
        goto release;
    }

    VkSemaphoreSubmitInfo waits[2] = {
        {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = vkf->sem[0],
            .value     = vkf->sem_value[0],
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        }, {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = g->out_sem,
            .value     = hold,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
    };
    VkSemaphoreSubmitInfo signals[2] = {
        {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = vkf->sem[0],
            .value     = vkf->sem_value[0] + 1,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        }, {
            .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = g->out_sem,
            .value     = hold + 1,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
    };
    VkCommandBufferSubmitInfo cmd_info = {
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = sl->cmd,
    };
    VkSubmitInfo2 submit = {
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount   = 2,
        .pWaitSemaphoreInfos      = waits,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmd_info,
        .signalSemaphoreInfoCount = 2,
        .pSignalSemaphoreInfos    = signals,
    };

    vkResetFences(g->vk->device, 1, &sl->fence);

    view_gpu_lock(g);
    ret = vkQueueSubmit2(g->queue, 1, &submit, sl->fence) == VK_SUCCESS ? 0 : CRAW_ERROR(EIO);
    view_gpu_unlock(g);

    if (ret >= 0)
        sl->busy = true;

    tm->submit += (double)(av_gettime_relative() - t0) / 1000.0;

    /* The contract the hwcontext documents: the value we signalled becomes
     * the value the next user has to wait on, and the layout is whatever we
     * left the image in. */
    if (ret >= 0) {
        vkf->sem_value[0] += 1;
        vkf->layout[0] = VK_IMAGE_LAYOUT_GENERAL;
        vkf->access[0] = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        if (vkf->queue_family[0] != VK_QUEUE_FAMILY_IGNORED)
            vkf->queue_family[0] = g->qf;
    }

    vkfc->unlock_frame(fc, vkf);

release:
    pl_vulkan_release_ex(g->gpu, pl_vulkan_release_params(
        .tex       = sl->tex,
        .layout    = VK_IMAGE_LAYOUT_GENERAL,
        .qf        = g->qf,
        /* Nothing was submitted on the error path, so the value libplacebo
         * waits on is the one it signalled itself. */
        .semaphore = (pl_vulkan_sem) { g->out_sem, ret >= 0 ? hold + 1 : hold },
    ));
    if (ret >= 0)
        g->out_value = hold + 1;

    if (ret < 0)
        return ret;

    g->display_slot = g->slot;
    g->slot = (g->slot + 1) % VIEW_SLOTS;
    g->submitted++;

    return 0;
}

int view_gpu_wait_prev(struct view_gpu *g, struct view_timing *tm)
{
    struct view_slot *sl = &g->slots[(g->display_slot + VIEW_SLOTS - 1) % VIEW_SLOTS];
    int64_t t0;

    if (g->submitted < 2 || !sl->busy)
        return 0;

    t0 = av_gettime_relative();
    if (vkWaitForFences(g->vk->device, 1, &sl->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return CRAW_ERROR(EIO);
    tm->fence += (double)(av_gettime_relative() - t0) / 1000.0;

    sl->busy = false;

    return 0;
}

/* ------------------------------------------------------- the video path */

void view_gpu_drop_src_views(struct view_gpu *g)
{
    if (!g->nb_src_views)
        return;

    vkDeviceWaitIdle(g->vk->device);
    for (int i = 0; i < g->nb_src_views; i++)
        vkDestroyImageView(g->vk->device, g->src_views[i].view, NULL);
    g->nb_src_views = 0;
}

void view_gpu_unmap(struct view_gpu *g)
{
    if (g->have_mapped) {
        pl_unmap_avframe(g->gpu, &g->mapped);
        g->have_mapped = false;
    }
}

int view_gpu_map(struct view_gpu *g, AVFrame *frame, AVStream *stream,
                 struct view_timing *tm)
{
    int64_t t0 = av_gettime_relative();
    /* The previous mapping is only dropped here, so that a paused window
     * can keep redrawing the picture it already has. */
    view_gpu_unmap(g);

    if (!pl_map_avframe_ex(g->gpu, &g->mapped, pl_avframe_params(
            .frame = frame,
            .tex   = g->map_tex,
        ))) {
        fprintf(stderr, "cannot map the decoded frame\n");
        return CRAW_ERROR(EIO);
    }

    /* libavcodec does not propagate stream level side data onto frames, so
     * anything the container said about colour is picked up here. The tags
     * are taken as they come: nothing is second-guessed. */
    pl_frame_copy_stream_props(&g->mapped, stream);

    g->have_mapped = true;
    g->width = (uint32_t)frame->width;
    g->height = (uint32_t)frame->height;
    tm->map += (double)(av_gettime_relative() - t0) / 1000.0;

    return 0;
}

float view_gpu_infer_ceiling(const struct view_gpu *g)
{
    struct pl_color_space csp;

    if (!g->have_mapped)
        return PL_COLOR_SDR_WHITE;

    csp = g->mapped.color;
    pl_color_space_infer(&csp);

    return csp.hdr.max_luma;
}

/* Exposure on the non-raw path has to happen in scene-linear light, which
 * is what libplacebo's LINEAR hook point is: after the transfer function has
 * been undone and before anything is scaled or tone mapped. The raw path
 * never uses this - libcraw applies exposure inside the colour stage. */
static const struct pl_hook *view_exposure_hook(struct view_gpu *g, float gain)
{
    char shader[512];
    int len;

    if (gain == 1.0f) {
        pl_mpv_user_shader_destroy(&g->exposure_hook);
        g->exposure_hook_gain = 1.0f;
        return NULL;
    }

    if (g->exposure_hook && g->exposure_hook_gain == gain)
        return g->exposure_hook;

    pl_mpv_user_shader_destroy(&g->exposure_hook);

    len = snprintf(shader, sizeof(shader),
        "//!HOOK LINEAR\n"
        "//!BIND HOOKED\n"
        "//!DESC craw-view exposure\n"
        "vec4 hook()\n"
        "{\n"
        "    vec4 color = HOOKED_texOff(0);\n"
        "    color.rgb *= %f;\n"
        "    return color;\n"
        "}\n", (double)gain);

    if (len < 0 || len >= (int)sizeof(shader))
        return NULL;

    g->exposure_hook = pl_mpv_user_shader_parse(g->gpu, shader, (size_t)len);
    g->exposure_hook_gain = gain;

    if (!g->exposure_hook)
        fprintf(stderr, "cannot build the exposure shader\n");

    return g->exposure_hook;
}

/* ------------------------------------------------------------ rendering */

void view_gpu_image(const struct view_gpu *g, struct pl_frame *out,
                    float ceiling)
{
    *out = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture           = g->slots[g->display_slot].tex,
            .components        = 3,
            .component_mapping = { 0, 1, 2, -1 },
        },
        .repr = {
            .sys    = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha  = PL_ALPHA_NONE,
        },
        .color = {
            .primaries = PL_COLOR_PRIM_BT_2020,
            /* Linear light with an absolute scale, which is what scRGB is.
             * Tagging it PL_COLOR_TRC_LINEAR instead would be equivalent
             * arithmetically - libplacebo neither scales nor clamps either
             * of them - but its peak detection is gated on
             * pl_color_transfer_nominal_peak(), which is 1.0 for LINEAR by
             * definition, so a linear-light source could never be measured
             * however high its hdr.max_luma. scRGB carries a nominal peak
             * of 10000 nits and passes that gate. The factor scRGB puts
             * between a sample and diffuse white is undone in
             * view_render_params(). */
            .transfer  = PL_COLOR_TRC_SCRGB,
            /* The pipeline writes scene-linear BT.2020 with diffuse white
             * at 1.0, so what the tone mapper is told about is the sensor
             * clip: the gain the colour stage applied, in nits. It is a
             * ceiling, not a measurement - peak detection finds the real
             * per-frame peak underneath it. */
            .hdr = {
                .min_luma = PL_COLOR_HDR_BLACK,
                .max_luma = ceiling,
            },
        },
        .crop = { 0, 0, (float)g->width, (float)g->height },
    };
}

static void view_render_params(struct view_state *st,
                               struct pl_render_params *rp,
                               struct pl_color_map_params *cmp,
                               struct pl_peak_detect_params *pdp,
                               struct pl_color_adjustment *adj,
                               const struct pl_hook **hook)
{
    *cmp = pl_color_map_default_params;
    cmp->tone_mapping_function = view_tone_map(st->look.tone_map);
    cmp->gamut_mapping = view_gamut_map(st->look.gamut_map);

    *pdp = pl_peak_detect_default_params;

    /* The raw path's texture is tagged scRGB, whose 1.0 is 80 nits, while
     * the pipeline's 1.0 is diffuse white at 203. Scaling the samples by
     * the ratio on the way in cancels exactly against the scale
     * libplacebo's scRGB linearization applies, so the numbers the tone
     * mapper sees are the pipeline's own. A mapped video frame carries its
     * own tags and wants none of this. */
    *adj = pl_color_adjustment_neutral;
    if (st->dec.raw)
        adj->contrast = PL_COLOR_SDR_WHITE / PL_COLOR_SCRGB_WHITE;

    *rp = pl_render_default_params;
    rp->color_adjustment = adj;
    rp->color_map_params = cmp;
    /* The static tag is the sensor clip, which almost no frame reaches;
     * measuring the real peak stops an ordinary frame from being tone
     * mapped as though it were full of blown highlights. libplacebo's
     * defaults for the smoothing window, the scene cut thresholds and the
     * percentile are kept: nothing measured here argues with them. */
    rp->peak_detect_params = st->look.peak_detect ? pdp : NULL;
    rp->lut = st->lut;
    rp->lut_type = st->look.lut_space > 0 ? PL_LUT_CONVERSION : st->look.lut_type;

    /* Only the non-raw path needs a shader for exposure; on the raw path
     * libcraw has already applied it, and the scRGB scale below would be
     * the wrong place for it anyway. */
    *hook = NULL;
    if (!st->dec.raw) {
        *hook = view_exposure_hook(&st->gpu, view_video_exposure(st));
        if (*hook) {
            rp->hooks = hook;
            rp->num_hooks = 1;
        }
    }
}

void view_gpu_reset_peak(struct view_gpu *g, bool export_renderer)
{
    /* The renderer holds the peak detection track (and the frame mixing
     * cache) between frames; a cut in the source has to clear it or the
     * first frames after it are tone mapped against the old scene. */
    pl_renderer_flush_cache(export_renderer ? g->rr_export : g->rr);
    if (!export_renderer)
        g->measured_peak = 0.0;
}

int view_gpu_display(struct view_gpu *g, struct view_state *st)
{
    int64_t t0 = av_gettime_relative();
    struct pl_swapchain_frame sw_frame;
    struct pl_render_params rp;
    struct pl_color_map_params cmp;
    struct pl_peak_detect_params pdp;
    struct pl_color_adjustment adj;
    struct pl_hdr_metadata measured;
    const struct pl_hook *hook;
    struct pl_frame image, target;
    struct pl_overlay overlay;
    struct pl_overlay_part part;

    if (!pl_swapchain_start_frame(g->sw, &sw_frame))
        return CRAW_ERROR(EAGAIN);

    pl_frame_from_swapchain(&target, &sw_frame);
    view_render_params(st, &rp, &cmp, &pdp, &adj, &hook);

    if (st->have_frame) {
        pl_rect2df src, dst;

        if (st->dec.raw)
            view_gpu_image(g, &image, view_peak_ceiling(st));
        else
            image = g->mapped;

        /* Zoom and pan are nothing but these two rectangles: the window of
         * source pixels to show and where on screen to put it. libplacebo
         * scales between them with its ordinary scaler, so at one-to-one it
         * samples a source pixel per output pixel and nothing is resampled
         * twice. */
        view_zoom_rects(st, &src, &dst);
        image.crop = src;
        target.crop = dst;
    }

    if (g->ui_tex[g->ui_slot]) {
        part = (struct pl_overlay_part) {
            .src = { 0, 0, (float)g->ui_w, (float)g->ui_h },
            .dst = { 0, 0, (float)g->ui_w, (float)g->ui_h },
        };
        overlay = (struct pl_overlay) {
            .tex    = g->ui_tex[g->ui_slot],
            .mode   = PL_OVERLAY_NORMAL,
            .coords = PL_OVERLAY_COORDS_DST_FRAME,
            .repr = {
                .sys    = PL_COLOR_SYSTEM_RGB,
                .levels = PL_COLOR_LEVELS_PC,
                /* nuklear draws into a cleared, fully transparent target
                 * with src-alpha blending, so what lands in the texture is
                 * colour already multiplied by coverage. */
                .alpha  = PL_ALPHA_PREMULTIPLIED,
            },
            .color     = pl_color_space_srgb,
            .parts     = &part,
            .num_parts = 1,
        };
        target.overlays = &overlay;
        target.num_overlays = 1;
    }

    if (!pl_render_image(g->rr, st->have_frame ? &image : NULL, &target, &rp)) {
        fprintf(stderr, "pl_render_image failed\n");
        return CRAW_ERROR(EIO);
    }

    /* Whatever the detector settled on for the frame just rendered. It is
     * reported in PQ, so it comes back through the same rescale libplacebo
     * uses internally. */
    if (st->look.peak_detect && pl_renderer_get_hdr_metadata(g->rr, &measured))
        g->measured_peak = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, measured.max_pq_y);
    else
        g->measured_peak = 0.0;

    if (!pl_swapchain_submit_frame(g->sw))
        return CRAW_ERROR(EIO);

    st->timing.render += (double)(av_gettime_relative() - t0) / 1000.0;
    t0 = av_gettime_relative();

    pl_swapchain_swap_buffers(g->sw);

    st->timing.present += (double)(av_gettime_relative() - t0) / 1000.0;

    return 0;
}

int view_gpu_render_export(struct view_gpu *g, struct view_state *st,
                           uint16_t *rgba)
{
    struct pl_render_params rp;
    struct pl_color_map_params cmp;
    struct pl_peak_detect_params pdp;
    struct pl_color_adjustment adj;
    const struct pl_hook *hook;
    struct pl_frame image, target;
    pl_fmt fmt;

    if (!g->export_tex || g->export_w != (int)g->width || g->export_h != (int)g->height) {
        pl_tex_destroy(g->gpu, &g->export_tex);

        fmt = pl_find_named_fmt(g->gpu, "rgba16");
        if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
            !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
            fprintf(stderr, "the GPU cannot render to a readable RGBA16 texture\n");
            return CRAW_ERROR(ENOTSUP);
        }

        g->export_tex = pl_tex_create(g->gpu, pl_tex_params(
            .w             = (int)g->width,
            .h             = (int)g->height,
            .format        = fmt,
            .renderable    = true,
            .storable      = !!(fmt->caps & PL_FMT_CAP_STORABLE),
            .blit_dst      = !!(fmt->caps & PL_FMT_CAP_BLITTABLE),
            .host_readable = true,
        ));
        if (!g->export_tex) {
            fprintf(stderr, "cannot create the %ux%u export texture\n", g->width, g->height);
            return CRAW_ERROR(ENOMEM);
        }
        g->export_w = (int)g->width;
        g->export_h = (int)g->height;
    }

    if (st->dec.raw)
        view_gpu_image(g, &image, view_peak_ceiling(st));
    else
        image = g->mapped;
    view_render_params(st, &rp, &cmp, &pdp, &adj, &hook);

    /* Display referred on purpose: what the PNG holds is the look on screen,
     * sRGB encoded, not the scene-linear numbers. */
    target = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture           = g->export_tex,
            .components        = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
        .repr = {
            .sys    = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha  = PL_ALPHA_NONE,
            .bits   = { .sample_depth = 16, .color_depth = 16 },
        },
        .color = pl_color_space_srgb,
        .crop  = { 0, 0, (float)g->width, (float)g->height },
    };

    /* The export has its own renderer so that the peak detection track it
     * walks through depends only on the exported frames: the same range
     * exported twice then gives the same files. */
    if (!pl_render_image(g->rr_export, &image, &target, &rp)) {
        fprintf(stderr, "pl_render_image failed while exporting\n");
        return CRAW_ERROR(EIO);
    }

    if (!pl_tex_download(g->gpu, pl_tex_transfer_params(
            .tex = g->export_tex,
            .ptr = rgba,
        ))) {
        fprintf(stderr, "cannot read the export texture back\n");
        return CRAW_ERROR(EIO);
    }

    return 0;
}

/* ------------------------------------------------------------- the UI tex */

void view_gpu_ui_drop_old(struct view_gpu *g)
{
    for (int i = 0; i < VIEW_SLOTS; i++) {
        if (g->ui_views_old[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(g->vk->device, g->ui_views_old[i], NULL);
            g->ui_views_old[i] = VK_NULL_HANDLE;
        }
        pl_tex_destroy(g->gpu, &g->ui_tex_old[i]);
    }
}

int view_gpu_ui_resize(struct view_gpu *g, int w, int h)
{
    VkFormat fmt;
    VkImageUsageFlags usage;
    VkImage image;
    pl_fmt plfmt;

    if (w <= 0 || h <= 0)
        return CRAW_ERROR(EINVAL);
    if (g->ui_tex[0] && g->ui_w == w && g->ui_h == h)
        return 0;

    vkDeviceWaitIdle(g->vk->device);

    /* nuklear's framebuffers still reference the current views, so they and
     * their images have to outlive nk_sdl_resize(). */
    view_gpu_ui_drop_old(g);
    for (int i = 0; i < VIEW_SLOTS; i++) {
        g->ui_views_old[i] = g->ui_views[i];
        g->ui_tex_old[i] = g->ui_tex[i];
        g->ui_views[i] = VK_NULL_HANDLE;
        g->ui_tex[i] = NULL;
    }

    plfmt = pl_find_named_fmt(g->gpu, "rgba8");
    if (!plfmt)
        return CRAW_ERROR(ENOTSUP);

    for (int i = 0; i < VIEW_SLOTS; i++) {
        g->ui_tex[i] = pl_tex_create(g->gpu, pl_tex_params(
            .w          = w,
            .h          = h,
            .format     = plfmt,
            .renderable = true,
            .sampleable = true,
        ));
        if (!g->ui_tex[i])
            return CRAW_ERROR(ENOMEM);

        image = pl_vulkan_unwrap(g->gpu, g->ui_tex[i], &fmt, &usage);
        if (!image)
            return CRAW_ERROR(ENOTSUP);

        VkImageViewCreateInfo info = {
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image    = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = fmt,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            },
        };
        if (vkCreateImageView(g->vk->device, &info, NULL, &g->ui_views[i]) != VK_SUCCESS)
            return CRAW_ERROR(ENOMEM);

        g->ui_format = fmt;
    }

    g->ui_slot = 0;
    g->ui_w = w;
    g->ui_h = h;

    return 0;
}
