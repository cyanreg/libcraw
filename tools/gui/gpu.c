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
#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/custom.h>
#include <libplacebo/tone_mapping.h>
/* Header-only in C, and it needs the Vulkan header first for AVVkFrame
 * support; view.h has already included it. */
#include <libplacebo/utils/libav.h>

#include "view.h"
#include "adjust.h"
#include "grain_ui.h"

/* BT.2408 diffuse white; a pipeline value of 1.0 is diffuse white, so the
 * sensor clip lands at this times the profile gain. */
#define VIEW_REF_WHITE_NITS 203.0f

/* --------------------------------------------------------------- tables */

/* What a .cube LUT expects at its input. Anything but the first entry makes
 * the LUT a PL_LUT_CONVERSION from that space to a BT.709 display, which is
 * how a camera or creative cube is meant to be applied. Supporting another
 * authoring space is one row here. */
static const struct pl_color_space view_lut_space_in[] = {
    { 0 },                                                          /* native */
    { .primaries = PL_COLOR_PRIM_BT_709,  .transfer = PL_COLOR_TRC_BT_1886 },
    { .primaries = PL_COLOR_PRIM_BT_2020, .transfer = PL_COLOR_TRC_HLG    },
    { .primaries = PL_COLOR_PRIM_BT_2020, .transfer = PL_COLOR_TRC_LINEAR },
    /* Panasonic publishes its V-Log to V-709 conversion as a cube taking
     * full-range V-Log code values in V-Gamut, which is what this names.
     * The output side is the BT.709 display every entry here converts to,
     * and V-709 is a BT.709 display encoding, so the two agree. */
    { .primaries = PL_COLOR_PRIM_V_GAMUT, .transfer = PL_COLOR_TRC_V_LOG   },
};

/* One word each: the same strings name the choice in the panel and on the
 * command line. */
const char *const view_lut_space_names[] = {
    "native", "bt709", "hlg2020", "linear2020", "vlog",
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

    /* The same again for the compute encoders the video export can use.
     * FFV1, APV and ProRes on Vulkan are ordinary compute shaders, and they
     * declare rather more than the decoders do: a runtime descriptor array
     * above all. This driver takes the modules anyway, but a validated
     * device refuses them, the pipeline comes back null, and the encode
     * writes a file with nothing in it - so the declaration has to be true
     * whether or not anyone is checking. What the device does not have,
     * libplacebo drops. The list is the one libavutil enables when it
     * creates a device of its own. */
    rq.required_features.shaderInt64 = VK_TRUE;
    rq.required_features.shaderStorageImageReadWithoutFormat  = VK_TRUE;
    rq.required_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    rq.required_features.shaderImageGatherExtended = VK_TRUE;
    rq.required_features_11.storageBuffer16BitAccess = VK_TRUE;
    rq.required_features_11.storagePushConstant16    = VK_TRUE;
    rq.required_features_11.samplerYcbcrConversion   = VK_TRUE;
    rq.required_features_12.runtimeDescriptorArray   = VK_TRUE;
    rq.required_features_12.shaderSampledImageArrayNonUniformIndexing  = VK_TRUE;
    rq.required_features_12.shaderStorageImageArrayNonUniformIndexing  = VK_TRUE;
    rq.required_features_12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    rq.required_features_12.shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
    rq.required_features_12.shaderInt8                   = VK_TRUE;
    rq.required_features_12.storageBuffer8BitAccess      = VK_TRUE;
    rq.required_features_12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    rq.required_features_12.storagePushConstant8         = VK_TRUE;
    rq.required_features_12.shaderFloat16                = VK_TRUE;
    rq.required_features_12.shaderBufferInt64Atomics     = VK_TRUE;
    rq.required_features_12.shaderSharedInt64Atomics     = VK_TRUE;
    rq.required_features_12.shaderSubgroupExtendedTypes  = VK_TRUE;
    rq.required_features_12.vulkanMemoryModel            = VK_TRUE;
    rq.required_features_12.vulkanMemoryModelDeviceScope = VK_TRUE;
    rq.required_features_13.computeFullSubgroups              = VK_TRUE;
    rq.required_features_13.subgroupSizeControl               = VK_TRUE;
    rq.required_features_13.shaderZeroInitializeWorkgroupMemory = VK_TRUE;

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
     * decoder take a path the device then rejects.
     *
     * The encode half of the list is the same story for the video export:
     * FFmpeg's Vulkan encoders refuse to start without the encode queue
     * extension and VK_KHR_video_maintenance1, and the latter does have a
     * feature, which is why it is chained below as well. */
    static const char *const video_exts[] = {
        VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME,
        VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,
        VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME,
#ifdef VK_KHR_video_encode_av1
        VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME,
#endif
        VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME,
#ifdef VK_KHR_video_maintenance2
        VK_KHR_VIDEO_MAINTENANCE_2_EXTENSION_NAME,
#endif
        /* Push descriptors became core in Vulkan 1.4, so libplacebo uses
         * them without asking for the extension - but FFmpeg decides which
         * descriptor path its compute shaders take by looking at the
         * enabled extension *names*. It has no feature of its own. */
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        /* The compute encoders' shaders declare SPV_KHR_maximal_reconvergence
         * and SPV_KHR_subgroup_rotate. Without the matching extensions and
         * their features a validated device refuses the modules, and
         * prores_ks_vulkan then writes a file of ninety kilobytes with
         * nothing in it. */
        VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME,
        VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME,
#ifdef VK_KHR_shader_relaxed_extended_instruction
        VK_KHR_SHADER_RELAXED_EXTENDED_INSTRUCTION_EXTENSION_NAME,
#endif
    };

    VkPhysicalDeviceShaderMaximalReconvergenceFeaturesKHR shader_mr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
        .pNext = &rq.required_features_11,
        .shaderMaximalReconvergence = VK_TRUE,
    };
    VkPhysicalDeviceVideoMaintenance1FeaturesKHR video_m1 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR,
        .pNext = &shader_mr,
        .videoMaintenance1 = VK_TRUE,
    };

    /* Promoted to Vulkan 1.4, so it goes in the core struct rather than in
     * one of its own; the extension above is only there for FFmpeg to see. */
    rq.required_features_14.shaderSubgroupRotate = VK_TRUE;

    feats.pNext = &video_m1;

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

    /* Which video encode engines this device has, for the export's preset
     * table: an engine is an extension, and libplacebo has just enabled
     * every one of the optional ones the device offers. A GPU without
     * VK_KHR_video_encode_av1 has no AV1 encoder, and the table greys the
     * family with that rather than finding out by opening one. */
    view_presets_device_exts(g->vk->extensions, g->vk->num_extensions);

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
    g->verbose = cfg->verbose;

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
            /* Eight bits per channel for an SDR window, even where the
             * surface offers ten. What the renderer dithers to is the
             * depth of this image, and a Wayland compositor does not put
             * this image on the screen: it composites it into a
             * framebuffer of its own, eight bits deep unless it was told
             * otherwise (sway's render_bit_depth, which defaults to 8),
             * rounding to it without dithering. A ten-bit window is then
             * dithered against a grid two bits finer than the one it
             * lands on, which leaves exactly the eight-bit contours the
             * dither is there to break: faint lines that follow the
             * gradients of a smooth picture. This asks for the depth that
             * is actually presented, so the dither meets it. Only the SDR
             * formats are given up - an HDR surface still gets ten bits,
             * and a compositor that accepts one presents them. */
            .disable_10bit_sdr = true,
        ));
        if (!g->sw) {
            fprintf(stderr, "cannot create the swapchain\n");
            return CRAW_ERROR(ENOTSUP);
        }
    }

    g->rr = pl_renderer_create(g->log, g->gpu);
    g->rr_compose = pl_renderer_create(g->log, g->gpu);
    g->rr_export = pl_renderer_create(g->log, g->gpu);
    if (!g->rr || !g->rr_compose || !g->rr_export)
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

    /* Before the context: the direct path owns a pipeline of its own, and a
     * command buffer out of the pool below. */
    view_gpu_direct_close(g);

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
        pl_tex_destroy(g->gpu, &g->linear_tex);
        pl_tex_destroy(g->gpu, &g->probe_tex);
        pl_buf_destroy(g->gpu, &g->probe_buf);
        for (int i = 0; i < VIEW_SLOTS; i++) {
            pl_tex_destroy(g->gpu, &g->slots[i].tex);
            pl_tex_destroy(g->gpu, &g->ui_tex[i]);
        }
        pl_tex_destroy(g->gpu, &g->export_tex);
        pl_tex_destroy(g->gpu, &g->pic_tex);
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
    pl_renderer_destroy(&g->rr_probe);
    free(g->probe_rgba);
    g->probe_rgba = NULL;
    g->probe_values = 0;
    g->probe_busy = false;
    pl_renderer_destroy(&g->rr_compose);
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
        /* The new pipeline has recorded nothing: the auto white balance
         * step keys its histogram read off this count, and left alone it
         * would ask the fresh pipeline for a frame that was never there. */
        g->submitted = 0;
    }

    if (!g->slots[0].tex || g->width != w || g->height != h) {
        plfmt = pl_find_named_fmt(g->gpu, "rgba16hf");
        if (!plfmt || !(plfmt->caps & PL_FMT_CAP_STORABLE) ||
            !(plfmt->caps & PL_FMT_CAP_HOST_READABLE)) {
            fprintf(stderr, "the GPU has no storable, readable RGBA16F format\n");
            return CRAW_ERROR(ENOTSUP);
        }

        for (int i = 0; i < VIEW_SLOTS; i++) {
            struct view_slot *sl = &g->slots[i];

            vkDestroyImageView(g->vk->device, sl->view, NULL);
            sl->view = VK_NULL_HANDLE;
            pl_tex_destroy(g->gpu, &sl->tex);
            sl->busy = false;

            /* Readable by the host as well, for the spot white balance,
             * which downloads a patch of it through view_gpu_read_region():
             * one more usage bit on the image, and the only alternative is
             * a second full-size texture and a blit. */
            sl->tex = pl_tex_create(g->gpu, pl_tex_params(
                .w             = (int)w,
                .h             = (int)h,
                .format        = plfmt,
                .storable      = true,
                .sampleable    = true,
                .host_readable = true,
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
    /* ACEScg, not linear BT.2020: DESIGN.md 8.1. BT.2020 does not contain
     * what this camera's matrix produces on a saturated light source, and
     * the negative components that leaves are what a display side gamut
     * mapper clips flat. AP1 holds them, and the ACES 1.3 reference gamut
     * compression at the end of the colour stage pulls the few that are
     * still outside back in. An export in another space asks the same
     * pipeline for it per frame through CRAWFrameParams.out_space. */
    pp.out_space  = CRAW_CS_ACES_AP1_LINEAR;
    pp.out_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    /* All of them always built: the panel can switch to auto white
     * balance, to denoising, to false colour suppression, to highlight
     * reconstruction, to defringe, to lens correction or to sharpening at
     * any time and a pipeline rebuild mid-clip would drop a frame. A
     * denoise pass that is built but left at strength 0 costs nothing
     * measurable, a false colour pass at zero passes, a defringe at
     * strength 0, a sharpening at zero iterations and an unflagged
     * highlight reconstruction are not dispatched, and a lens pass handed a
     * frame with no record drops out entirely, which is what makes them all
     * live controls. The temporal denoise is the same at 0, and its state,
     * 175 MB at 5.8K, is the price of being able to turn it on without
     * losing a frame. The sharpening also changes its iteration count
     * between playing and paused, which changes how many dispatches are
     * recorded and nothing that was built. */
    pp.flags      = CRAW_PIPELINE_FLAG_AWB_HISTOGRAM | CRAW_PIPELINE_FLAG_DENOISE |
                    CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE |
                    CRAW_PIPELINE_FLAG_FALSE_COLOR | CRAW_PIPELINE_FLAG_CHROMA_DENOISE |
                    CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION | CRAW_PIPELINE_FLAG_DEFRINGE |
                    CRAW_PIPELINE_FLAG_LENS | CRAW_PIPELINE_FLAG_SHARPEN;
    /* Enough copies that the previous frame's histogram is still readable
     * while the current one is in flight, which is what lets the auto white
     * balance run without a per-frame stall. */
    pp.nb_awb_slots = 3;

    ret = craw_pipeline_create(g->ctx, &pp, &g->pipe);
    if (ret < 0) {
        fprintf(stderr, "craw_pipeline_create(%ux%u) failed: %d\n", w, h, ret);
        return ret;
    }

    if (g->verbose)
        fprintf(stderr, "pipeline: built for %ux%u, %s\n", w, h,
                (unsigned)debayer < CRAW_DEBAYER_NB
                    ? view_debayer_names[debayer] : "?");

    g->width = w;
    g->height = h;
    g->pipe_debayer = debayer;

    return 0;
}

/* The clip that is going away, while its decoder is still alive: an image
 * view must not outlive its image, and everything below points at frames
 * the decoder is about to free.
 *
 *  - the cached views of its frame pool, because the next decoder
 *    allocates a pool of its own and a recycled VkImage handle would
 *    otherwise find a view of an image that no longer exists;
 *  - the developed path's mapping, which holds one of its frames;
 *  - the slot bookkeeping, because the auto white balance reads back the
 *    histogram of submission N - 1 and the next clip has submitted none;
 *  - the peak detection track in both renderers, since two clips are a cut
 *    and the old scene's peak must not tone map the new one's first
 *    frames. */
void view_gpu_release_frames(struct view_gpu *g)
{
    if (!g->vk)
        return;

    /* Every fence and every command buffer is retired before the images
     * they were recorded against go. */
    vkDeviceWaitIdle(g->vk->device);

    view_gpu_drop_src_views(g);
    view_gpu_unmap(g);
    /* The next clip may be another size or already developed, and the
     * direct path's pipeline is built for exactly one size; an export that
     * survives a clip switch is not a thing anyway. */
    view_gpu_direct_close(g);

    for (int i = 0; i < VIEW_SLOTS; i++)
        g->slots[i].busy = false;
    g->slot = g->display_slot = 0;
    g->submitted = 0;
    g->last_gpu_ms = 0.0;

    view_gpu_reset_peak(g, false);
    view_gpu_reset_peak(g, true);
}

/* And what the clip that is arriving needs, once its kind is known. A
 * developed clip can use neither the pipeline nor its full-size output
 * textures, and on a 5.8K clip that is several hundred megabytes to sit on
 * for nothing; two raw clips of the same size keep both and
 * view_gpu_ensure() then has nothing to do at all. */
void view_gpu_new_clip(struct view_gpu *g, bool raw)
{
    if (!g->vk || raw || !g->pipe)
        return;

    craw_pipeline_free(&g->pipe);

    for (int i = 0; i < VIEW_SLOTS; i++) {
        vkDestroyImageView(g->vk->device, g->slots[i].view, NULL);
        g->slots[i].view = VK_NULL_HANDLE;
        pl_tex_destroy(g->gpu, &g->slots[i].tex);
    }

    if (g->verbose)
        fprintf(stderr, "pipeline: released, %ux%u, the clip is developed\n",
                g->width, g->height);

    g->width = g->height = 0;
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

/* The raw path's one submission, for a picture (view_gpu_process()) or for
 * a pre-roll frame (view_gpu_preroll()). */
static int view_gpu_develop(struct view_gpu *g, const AVFrame *frame,
                            const CRAWFrameParams *fp, struct view_timing *tm,
                            bool preroll)
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
                        "plane; cineview needs an RGGB Bayer mosaic\n");
        return CRAW_ERROR(ENOTSUP);
    }

    src_view = view_src_view(g, vkf->img[0], vkfmt[0]);
    if (src_view == VK_NULL_HANDLE)
        return CRAW_ERROR(ENOMEM);

    /* The only fence this waits on belongs to the slot about to be reused,
     * which was submitted VIEW_SLOTS frames ago and has normally long since
     * retired. Frame N is never waited for before N + 1 is recorded. A
     * pre-roll is the exception: its frames all go into this one slot, so
     * each waits for the one before it, and the picture after them for the
     * last. */
    if (sl->busy) {
        int64_t tw = av_gettime_relative();

        if (vkWaitForFences(g->vk->device, 1, &sl->fence, VK_TRUE,
                            UINT64_MAX) != VK_SUCCESS)
            return CRAW_ERROR(EIO);

        if (!preroll) {
            tm->fence += (double)(av_gettime_relative() - tw) / 1000.0;
            if (g->rec)
                g->rec->fence_ms += (double)(av_gettime_relative() - tw) / 1000.0;
        }
        sl->busy = false;

        /* A pre-roll frame writes no timestamps, so a slot it last used has
         * none to give and the averages below are left alone. */
        if (vkGetQueryPoolResults(g->vk->device, g->queries, (uint32_t)(2 * g->slot),
                                  2, sizeof(ts), ts, sizeof(ts[0]),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            g->last_gpu_ms = (double)(ts[1] - ts[0]) * g->timestamp_period / 1.0e6;
            /* This slot's flag still describes the submission the
             * timestamps came from; it is overwritten below for the one
             * about to be recorded. */
            g->last_gpu_denoise = g->slot_denoise[g->slot];
            g->last_gpu_lens = g->slot_lens[g->slot];
        }
    }

    /* The decoder's image whole: it was told not to crop (decode.c), so
     * this is the camera's entire readout and the pipeline develops all of
     * it from its own (0, 0), which is the sensel the profile's CFA phase
     * describes. What the frame header marked as the picture is the
     * viewer's crop rectangle, and that is a view of the result. */
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

    /* Charged to this slot's timestamps, which are read back the next time
     * the slot comes round. */
    g->slot_denoise[g->slot] = fp->denoise_strength > 0.0f ||
                               fp->denoise_temporal_strength > 0.0f;
    g->slot_lens[g->slot] = fp->lens.valid;

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

    /* Not for a pre-roll frame: the panel's pass times are of the
     * pictures, and a pre-roll frame is developed without the stages after
     * the raw one (main.c). */
    if (!preroll)
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

    if (!preroll)
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

    if (!preroll) {
        tm->submit += (double)(av_gettime_relative() - t0) / 1000.0;
        if (g->rec)
            g->rec->t_submit = av_gettime_relative();
    }

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

    /* A pre-roll frame is not a picture: the slot stays the next one to
     * record into, the one on screen stays on screen, and nothing that
     * counts pictures sees it. */
    if (preroll)
        return 0;

    g->display_slot = g->slot;
    g->slot = (g->slot + 1) % VIEW_SLOTS;
    g->submitted++;
    g->pic_serial++;

    return 0;
}

int view_gpu_process(struct view_gpu *g, const AVFrame *frame,
                     const CRAWFrameParams *fp, struct view_timing *tm)
{
    return view_gpu_develop(g, frame, fp, tm, false);
}

int view_gpu_preroll(struct view_gpu *g, const AVFrame *frame,
                     const CRAWFrameParams *fp)
{
    return view_gpu_develop(g, frame, fp, NULL, true);
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
    if (g->rec)
        g->rec->fence_ms += (double)(av_gettime_relative() - t0) / 1000.0;

    sl->busy = false;

    return 0;
}

int view_gpu_read_region(struct view_gpu *g, int x, int y, int w, int h,
                         uint16_t *rgba)
{
    struct view_slot *sl = &g->slots[g->display_slot];

    if (!g->pipe || !sl->tex || !g->submitted || !rgba)
        return CRAW_ERROR(EINVAL);
    if (x < 0 || y < 0 || w <= 0 || h <= 0 ||
        (uint32_t)(x + w) > g->width || (uint32_t)(y + h) > g->height)
        return CRAW_ERROR(EINVAL);

    /* The slot's own fence, so that the read is of the pass that was
     * submitted and not of whatever the texture held before it. `busy` is
     * left set: the next recording into this slot reads its timestamps
     * through that flag, and a wait that has already been paid for
     * returns at once. libplacebo orders its copy behind the semaphore the
     * slot was released with as well, so this is belt and braces on the
     * device and a plain statement of intent on the host. */
    if (sl->busy &&
        vkWaitForFences(g->vk->device, 1, &sl->fence, VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS)
        return CRAW_ERROR(EIO);

    /* The rectangle and nothing else: no row pitch means the rows come
     * out packed at the rectangle's own width, and a download to host
     * memory without a callback blocks until the copy has landed. */
    if (!pl_tex_download(g->gpu, pl_tex_transfer_params(
            .tex = sl->tex,
            .rc  = { x, y, 0, x + w, y + h, 1 },
            .ptr = rgba,
        ))) {
        fprintf(stderr, "cannot read the picture back\n");
        return CRAW_ERROR(EIO);
    }

    return 0;
}

/* The mapped picture's colour as the window's render resolves it. For most
 * sources that is only the gaps filled in, but an HLG signal is display
 * relative: libplacebo takes its black level from the target, and the
 * black lift that decides - applied per channel, in the signal - moves the
 * ratios of everything dark. Reading the linear light back against any
 * other target would measure a picture the LINEAR stage never holds. The
 * window's target when there is a window, sRGB otherwise, which is what a
 * still is written in; a video exported to an HDR space resolves an HLG
 * source a little differently again, and is corrected as the window is. */
static struct pl_color_space view_gpu_source_color(const struct view_gpu *g)
{
    struct pl_color_space src = g->mapped.color;
    struct pl_color_space ref = g->pic_have_target ? g->pic_color
                                                   : pl_color_space_srgb;

    pl_color_space_infer_map(&src, &ref);

    return src;
}

bool view_gpu_source_space(const struct view_gpu *g,
                           enum pl_color_primaries *prim, float *peak)
{
    struct pl_color_space csp;

    if (!g->have_mapped)
        return false;

    csp = view_gpu_source_color(g);

    if (prim)
        *prim = csp.primaries;
    if (peak)
        *peak = csp.hdr.max_luma / PL_COLOR_SDR_WHITE;

    return true;
}

/* One render of the mapped picture, or the `crop` of it, into `tex` as
 * linear light in the picture's own primaries. The source goes in resolved
 * the way the window resolves it (view_gpu_source_color()), and the target
 * carries that same luminance range, so nothing is tone mapped, and the
 * same primaries, so nothing is gamut mapped: what lands is exactly what
 * the LINEAR stage holds, which is where the white balance is applied. No
 * hooks, no peak detection, no dither. */
static bool view_gpu_linear_pass(struct view_gpu *g, const pl_rect2df *crop,
                                 pl_tex tex)
{
    struct pl_render_params rp = pl_render_default_params;
    struct pl_color_space csp = view_gpu_source_color(g);
    struct pl_frame image = g->mapped, target;

    if (!g->rr_probe)
        g->rr_probe = pl_renderer_create(g->log, g->gpu);
    if (!g->rr_probe)
        return false;

    image.color = csp;
    if (crop)
        image.crop = *crop;

    target = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture           = tex,
            .components        = 3,
            .component_mapping = { 0, 1, 2, -1 },
        },
        .repr = {
            .sys    = PL_COLOR_SYSTEM_RGB,
            .levels = PL_COLOR_LEVELS_PC,
            .alpha  = PL_ALPHA_NONE,
        },
        .color = {
            .primaries = csp.primaries,
            .transfer  = PL_COLOR_TRC_LINEAR,
            .hdr       = csp.hdr,
        },
    };

    rp.peak_detect_params = NULL;
    rp.dither_params = NULL;

    return pl_render_image(g->rr_probe, &image, &target, &rp);
}

/* A float texture the probe renderer can draw into and the host can read
 * back, kept at the size it was last asked for. */
static bool view_gpu_float_tex(struct view_gpu *g, pl_tex *tex, int w, int h)
{
    pl_fmt fmt;

    if (*tex && (*tex)->params.w == w && (*tex)->params.h == h)
        return true;

    pl_tex_destroy(g->gpu, tex);

    fmt = pl_find_named_fmt(g->gpu, "rgba32f");
    if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
        !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        fprintf(stderr, "the GPU cannot render to a readable RGBA32F texture\n");
        return false;
    }

    *tex = pl_tex_create(g->gpu, pl_tex_params(
        .w = w,
        .h = h,
        .format = fmt,
        .renderable = true,
        .host_readable = true,
    ));

    return *tex != NULL;
}

int view_gpu_read_linear(struct view_gpu *g, int x, int y, int w, int h,
                         float *rgba)
{
    pl_rect2df crop = { (float)x, (float)y, (float)(x + w), (float)(y + h) };

    if (!g->have_mapped || !rgba || w <= 0 || h <= 0 || x < 0 || y < 0 ||
        (uint32_t)(x + w) > g->width || (uint32_t)(y + h) > g->height)
        return CRAW_ERROR(EINVAL);

    if (!view_gpu_float_tex(g, &g->linear_tex, w, h))
        return CRAW_ERROR(ENOTSUP);

    if (!view_gpu_linear_pass(g, &crop, g->linear_tex) ||
        !pl_tex_download(g->gpu, pl_tex_transfer_params(
            .tex = g->linear_tex,
            .ptr = rgba,
        ))) {
        fprintf(stderr, "cannot read the picture back in linear light\n");
        return CRAW_ERROR(EIO);
    }

    return 0;
}

int view_gpu_probe_start(struct view_gpu *g)
{
    double scale;
    int w, h;

    if (!g->have_mapped || !g->width || !g->height)
        return CRAW_ERROR(EINVAL);
    if (g->probe_busy)
        return CRAW_ERROR(EAGAIN);

    scale = fmin(1.0, fmin(960.0 / g->width, 540.0 / g->height));
    w = CRAW_MAX(1, (int)lrint(g->width * scale));
    h = CRAW_MAX(1, (int)lrint(g->height * scale));

    if (!view_gpu_float_tex(g, &g->probe_tex, w, h))
        return CRAW_ERROR(ENOTSUP);

    if (!g->probe_buf || g->probe_buf->params.size != (size_t)w * h * 16) {
        pl_buf_destroy(g->gpu, &g->probe_buf);
        g->probe_buf = pl_buf_create(g->gpu, pl_buf_params(
            .size = (size_t)w * h * 16,
            .host_readable = true,
        ));
        if (!g->probe_buf)
            return CRAW_ERROR(ENOMEM);
    }

    if (!view_gpu_linear_pass(g, NULL, g->probe_tex) ||
        !pl_tex_download(g->gpu, pl_tex_transfer_params(
            .tex = g->probe_tex,
            .buf = g->probe_buf,
        )))
        return CRAW_ERROR(EIO);

    g->probe_w = w;
    g->probe_h = h;
    g->probe_busy = true;

    return 0;
}

int view_gpu_probe_poll(struct view_gpu *g, bool wait, float **rgba,
                        int *w, int *h)
{
    size_t values;

    if (!g->probe_busy ||
        pl_buf_poll(g->gpu, g->probe_buf, wait ? UINT64_MAX : 0))
        return 0;

    g->probe_busy = false;

    values = (size_t)g->probe_w * g->probe_h * 4;
    if (values > g->probe_values) {
        float *p = realloc(g->probe_rgba, values * sizeof(*p));

        if (!p)
            return CRAW_ERROR(ENOMEM);
        g->probe_rgba = p;
        g->probe_values = values;
    }

    if (!pl_buf_read(g->gpu, g->probe_buf, 0, g->probe_rgba,
                     values * sizeof(float)))
        return CRAW_ERROR(EIO);

    *rgba = g->probe_rgba;
    *w = g->probe_w;
    *h = g->probe_h;

    return 1;
}

/* Asks, without blocking, whether the picture just submitted has finished
 * on the GPU, and records the first iteration at which it had. Only --trace
 * calls it, and only so that the file can say whether libcraw's work was
 * done before the render that needed it was submitted; the resolution is
 * therefore one iteration of the run loop. */
void view_gpu_poll_fence(struct view_gpu *g)
{
    struct view_slot *sl = &g->slots[g->display_slot];

    if (!g->rec || g->rec->t_fence || !g->submitted || !sl->busy)
        return;

    if (vkGetFenceStatus(g->vk->device, sl->fence) == VK_SUCCESS)
        g->rec->t_fence = av_gettime_relative();
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
                 bool vlog, struct view_timing *tm)
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

    /* Except for the one clip whose tags are known to be wrong. A V-Log
     * recording from this camera is written with a BT.709 VUI throughout,
     * and only the Y'CbCr matrix of it is true: the camera really does
     * encode the V-Log signal through the BT.709 matrix at full range, and
     * decoding it as anything else tilts the colour of the whole picture.
     * So the two tags that describe the light are replaced and everything
     * else - the matrix, the range, the chroma siting - is left exactly as
     * the stream declared it. Downstream nothing changes: libplacebo
     * linearises V-Log, measures the peak, tone maps to the display, and
     * the exposure hook, the LUT, the adjustments and the grain all run
     * where they always did. */
    if (vlog) {
        g->mapped.color.primaries = PL_COLOR_PRIM_V_GAMUT;
        g->mapped.color.transfer  = PL_COLOR_TRC_V_LOG;
    }

    g->have_mapped = true;
    g->width = (uint32_t)frame->width;
    g->height = (uint32_t)frame->height;
    g->pic_serial++;
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
            /* The pipeline's hand-off space, DESIGN.md 8.1. libplacebo's
             * AP1 is the same primaries libcraw builds the matrix from -
             * R(0.713,0.293) G(0.165,0.830) B(0.128,0.044) - with the ACES
             * white (0.32168, 0.33767), checked against its own table, so
             * the two agree about what an AP1 sample means. libplacebo then
             * adapts that white to the display's with CAT16 rather than the
             * Bradford libcraw used to get here; a von Kries transform maps
             * the source white onto the destination white by construction,
             * so neutrals come back exactly and only chromatic colours
             * differ. */
            .primaries = PL_COLOR_PRIM_ACES_AP1,
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
            /* The pipeline writes scene-linear ACEScg with diffuse white
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

/* The hook state for one of the two renderers, made on first use the way
 * the grain's is: the pl_gpu has to exist first, and a viewer that never
 * exports never builds the export one. */
static craw_adjust *view_adjust_state(struct view_state *st, bool export_path)
{
    craw_adjust **slot = export_path ? &st->adjustments.state_export
                                     : &st->adjustments.state;

    if (!*slot && st->gpu.gpu)
        *slot = craw_adjust_create(st->gpu.gpu);

    return *slot;
}

/* FNV-1a. What it is hashing is the display path's inputs, so that the
 * window can tell whether the picture it already has is still the right
 * one; nothing is stored or compared across runs. */
static uint64_t view_hash(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = p;

    while (n--) {
        h ^= *b++;
        h *= 0x100000001b3ULL;
    }

    return h;
}

/* ------------------------------------------------- the tone map list */

/* libplacebo's own table of curves, by index, so that the panel's combo,
 * --tone-map and the settings line all name exactly what the library has
 * and nothing is typed into this program. */
int view_tone_map_count(void)
{
    return pl_num_tone_map_functions;
}

const char *view_tone_map_name(int i)
{
    if (i < 0 || i >= pl_num_tone_map_functions)
        return "?";

    return pl_tone_map_functions[i]->name;
}

int view_tone_map_default(void)
{
    for (int i = 0; i < pl_num_tone_map_functions; i++)
        if (pl_tone_map_functions[i] ==
            pl_color_map_default_params.tone_mapping_function)
            return i;

    return 0;
}

int view_tone_map_find(const char *name)
{
    const struct pl_tone_map_function *f = pl_find_tone_map_function(name);

    for (int i = 0; f && i < pl_num_tone_map_functions; i++)
        if (pl_tone_map_functions[i] == f)
            return i;

    return -1;
}

static void view_render_params(struct view_state *st,
                               struct pl_render_params *rp,
                               struct pl_color_map_params *cmp,
                               struct pl_peak_detect_params *pdp,
                               struct pl_color_adjustment *adj,
                               const struct pl_hook **hook, bool export_path,
                               uint64_t *sig)
{
    /* The look is bypassed entirely while a scene-referred export space is
     * selected, on screen as well as in the file: what that export holds is
     * the pipeline's own linear light, so the window has to show the same
     * picture up to the display transform or the panel would be grading
     * something nobody is going to get. */
    bool scene = view_export_is_scene(st);
    int tm = CRAW_CLAMP(st->look.tone_map, 0, pl_num_tone_map_functions - 1);

    /* One mapping for the window and the file: the tone map the Look
     * section picks, spline unless it is told otherwise, and libplacebo's
     * own defaults for the rest - the perceptual gamut map, the peak
     * detection - into whatever the target is, which for the window is
     * what the compositor negotiated and for an export is the colour space
     * that was chosen. The export used to have a mapping of its own,
     * bt.2390 by default, and a file that came out thirteen per cent
     * brighter in the mid-tones than the frame on screen is what that
     * bought; a display-referred export is the picture as the viewer shows
     * it, and setting the curve here, for both, is what makes it so. */
    *cmp = pl_color_map_default_params;
    cmp->tone_mapping_function = pl_tone_map_functions[tm];
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
    rp->peak_detect_params = pdp;
    if (!export_path) {
        /* The window lets the measurement arrive a frame late. Without this
         * libplacebo reads the peak back inside pl_render_image and has to
         * wait for the GPU to get there, which on one queue means waiting
         * for the decoder's and libcraw's work as well: a full round trip
         * per displayed frame, measured at ten milliseconds on this
         * machine. An export reads its peak the way it always did - nothing
         * about a written file may depend on when the CPU asked. */
        pdp->allow_delayed = true;
    }
    /* A user .cube goes on the image as asked; a film simulation never
     * does: it is a display-referred sRGB look and belongs after tone
     * mapping, in the output's own encoding. It used to ride on the target
     * frame, which libplacebo applies at the very end of its output pass -
     * after every PL_HOOK_PRE_OUTPUT hook - so nothing could ever be put in
     * front of it. It is now the first of the adjustment hooks below, on
     * the same values and through the same pl_shader_custom_lut, so that
     * vibrance and soft light can follow it the way they follow film
     * simulation in RawTherapee.
     *
     * The Look section's switch takes both away at once and leaves the
     * selection selected, which is what view_look_active() answers.
     *
     * The menu's preview stands in for the chosen film on the window, and
     * shows as the rendering would once picked: the hook attached whatever
     * the section's switch says, since picking switches it on, and in
     * place of a user .cube, since picking replaces it. An export renders
     * the choice. */
    const struct pl_custom_lut *film =
        !export_path && st->preview ? st->preview
      : view_look_active(st) && st->look.film_sim >= 0 ? st->lut : NULL;

    rp->lut = !film && st->look.film_sim < 0 && view_look_active(st) ? st->lut : NULL;
    rp->lut_type = st->look.lut_space > 0 ? PL_LUT_CONVERSION : st->look.lut_type;

    /* Only the non-raw path has anything to do in linear light: the white
     * balance and the exposure, as one matrix at libplacebo's LINEAR stage
     * - after the transfer function is undone and before anything is
     * scaled or tone mapped. On the raw path libcraw has already applied
     * both, and the scRGB scale below would be the wrong place for either
     * anyway. The matrix is a uniform of the hook, so it is what the
     * signature below hashes, and an identity is not attached at all. */
    int nb_hooks = 0;
    pl_matrix3x3 linear = pl_matrix3x3_identity;
    bool have_linear = false;
    if (!st->dec.raw) {
        craw_adjust *aj = view_adjust_state(st, export_path);
        const struct pl_hook *h;

        have_linear = view_vwb_linear(st, &linear);
        if (have_linear && aj && (h = craw_adjust_linear(aj, &linear)))
            hook[nb_hooks++] = h;
    }

    /* The display-referred chain, all four of it at PL_HOOK_PRE_OUTPUT, so
     * everything here lands on the picture after scaling and colour
     * management: what is graded is what is on screen, at the size it is on
     * screen. libplacebo runs the hooks of one stage in the order of this
     * array, so the array is the order, and the order is RawTherapee's -
     * film simulation in its RGB stage, vibrance in the Lab stage after it,
     * soft light late in that same stage - with the grain last because
     * grain is a property of the medium and everything above it is a
     * property of the grade.
     *
     * Each hook is left off entirely when its parameters would make it an
     * identity, so the defaults cost nothing and change nothing.
     *
     * The export path gets its own shader state throughout - and, for the
     * grain, its own frame-index seed - so exporting a range twice gives
     * the same files. */
    if (!scene) {
        craw_adjust *aj = view_adjust_state(st, export_path);
        const struct pl_hook *h;

        if (aj && film && (h = craw_adjust_lut(aj, film)))
            hook[nb_hooks++] = h;

        /* Soft light and vibrance are one switch of their own: the
         * Adjustments section, off, leaves both hooks unattached at
         * whatever amounts they are set to. */
        if (aj && view_adjust_active(st)) {
            const struct craw_vibrance_params vp = {
                .pastels      = st->adjustments.pastels,
                .saturated    = st->adjustments.saturated,
                .threshold    = st->adjustments.threshold,
                .protect_skin = st->adjustments.protect_skin,
                .avoid_shift  = st->adjustments.avoid_shift,
            };

            if ((h = craw_adjust_vibrance(aj, &vp)))
                hook[nb_hooks++] = h;

            if ((h = craw_adjust_softlight(aj, st->adjustments.softlight)))
                hook[nb_hooks++] = h;
        }

        if ((h = view_grain_hook(st, export_path)))
            hook[nb_hooks++] = h;
    }

    if (nb_hooks) {
        rp->hooks = hook;
        rp->num_hooks = nb_hooks;
    }

    if (!sig)
        return;

    /* Everything resolved above, in one number. The display path renders
     * only when this changes, so a look parameter added to this function
     * has to be added here too or a paused picture will not follow it: the
     * hooks are cached objects whose address does not move when their
     * parameters do, which is why the parameters themselves are hashed and
     * not just the pointers. Each of the three structs is hashed up to its
     * trailing GPU state, which is not a look. */
    *sig = view_hash(0xcbf29ce484222325ULL, cmp, sizeof(*cmp));
    *sig = view_hash(*sig, pdp, sizeof(*pdp));
    *sig = view_hash(*sig, adj, sizeof(*adj));
    *sig = view_hash(*sig, &rp->lut, sizeof(rp->lut));
    *sig = view_hash(*sig, &rp->lut_type, sizeof(rp->lut_type));
    /* The film hook is one object whichever LUT it carries, so the LUT
     * itself is what tells a preview from the choice - by its signature,
     * libplacebo's hash of the cube, and not by its address: one preview
     * freed and the next parsed lands on the same allocation, and a paused
     * picture hashed on the pointer stayed on the first rendering hovered. */
    uint64_t film_sig = film ? film->signature : 0;
    *sig = view_hash(*sig, &film_sig, sizeof(film_sig));
    *sig = view_hash(*sig, &nb_hooks, sizeof(nb_hooks));
    *sig = view_hash(*sig, hook, (size_t)nb_hooks * sizeof(hook[0]));
    *sig = view_hash(*sig, &st->look, offsetof(struct view_look, video_preset));
    *sig = view_hash(*sig, &st->look.video_preset,
                     sizeof(st->look.video_preset));
    *sig = view_hash(*sig, &st->adjustments,
                     offsetof(struct view_adjust, state));
    *sig = view_hash(*sig, &st->grain, offsetof(struct view_grain, hook));
    *sig = view_hash(*sig, &have_linear, sizeof(have_linear));
    *sig = view_hash(*sig, &linear, sizeof(linear));
    *sig = view_hash(*sig, &scene, sizeof(scene));
}

void view_gpu_reset_peak(struct view_gpu *g, bool export_renderer)
{
    /* The renderer holds the peak detection track (and the frame mixing
     * cache) between frames; a cut in the source has to clear it or the
     * first frames after it are tone mapped against the old scene. */
    pl_renderer_flush_cache(export_renderer ? g->rr_export : g->rr);
    if (!export_renderer) {
        g->measured_peak = 0.0;
        /* The kept picture was tone mapped against the track that has just
         * been thrown away, so it is no longer what this frame looks like. */
        g->pic_valid = false;
    }
}

/* The texture the window's picture is kept in between renders: the size of
 * the swapchain image and floating point, so that the picture pass does no
 * quantisation of its own and the compose below is the only place the
 * display's bit depth is met. False when there is no such format or no
 * memory, which makes the caller render straight to the swapchain the way
 * it always did. */
static bool view_gpu_pic_tex(struct view_gpu *g, int w, int h)
{
    pl_fmt fmt;

    if (g->pic_tex && g->pic_w == w && g->pic_h == h)
        return true;

    pl_tex_destroy(g->gpu, &g->pic_tex);
    g->pic_valid = false;
    g->pic_w = g->pic_h = 0;

    fmt = pl_find_fmt(g->gpu, PL_FMT_FLOAT, 4, 16, 0,
                      PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_SAMPLEABLE |
                      PL_FMT_CAP_LINEAR);
    if (!fmt)
        return false;

    /* Blittable and storable where the format allows it: libplacebo picks
     * a blit or a compute pass over a render pass when the work is a plain
     * copy, which both of the passes either side of this texture often are,
     * and a texture that cannot take one makes it log the refusal on every
     * frame. */
    g->pic_tex = pl_tex_create(g->gpu, pl_tex_params(
        .w          = w,
        .h          = h,
        .format     = fmt,
        .renderable = true,
        .sampleable = true,
        .storable   = !!(fmt->caps & PL_FMT_CAP_STORABLE),
        .blit_src   = !!(fmt->caps & PL_FMT_CAP_BLITTABLE),
        .blit_dst   = !!(fmt->caps & PL_FMT_CAP_BLITTABLE),
    ));
    if (!g->pic_tex)
        return false;

    g->pic_w = w;
    g->pic_h = h;

    return true;
}

/* The picture pass: everything the window shows except the panel, rendered
 * into pic_tex against the description the swapchain last carried. Returns
 * true when it actually rendered; false means the texture already holds
 * exactly this picture and nothing was done.
 *
 * It is split out from the present so that it can be run the moment a
 * picture has been developed rather than at the moment the picture is due:
 * pl_render_image blocks for five to twenty-five milliseconds here - the
 * CPU waiting its turn on the one queue the decoder, libcraw and the
 * renderer all share - and on the present path every one of those
 * milliseconds lands in the interval between one picture and the next. */
static bool view_gpu_pic_pass(struct view_gpu *g, struct view_state *st,
                              const struct pl_color_repr *trepr,
                              const struct pl_color_space *tcolor,
                              int tw, int th)
{
    struct pl_render_params rp;
    struct pl_color_map_params cmp;
    struct pl_peak_detect_params pdp;
    struct pl_color_adjustment adj;
    struct pl_hdr_metadata measured;
    const struct pl_hook *hook[VIEW_MAX_HOOKS];
    struct pl_frame image, pic;
    pl_rect2df dst = { 0, 0, (float)tw, (float)th };
    uint64_t sig;
    float ceiling;

    if (!view_gpu_pic_tex(g, tw, th))
        return false;

    view_render_params(st, &rp, &cmp, &pdp, &adj, hook, false, &sig);
    if (st->trace.on)
        rp.info_callback = view_trace_pass;

    ceiling = st->have_frame ? view_peak_ceiling(st) : 0.0f;
    if (st->have_frame) {
        pl_rect2df src;

        if (st->dec.raw)
            view_gpu_image(g, &image, ceiling);
        else
            image = g->mapped;

        /* Zoom and pan are nothing but these two rectangles: the window of
         * the picture to show and where on screen to put it. libplacebo
         * scales between them with its ordinary scaler, so at one-to-one it
         * samples a source pixel per output pixel and nothing is resampled
         * twice.
         *
         * The first is in the rotated picture's pixels and libplacebo wants
         * the source's, so it turns back here and the renderer is told to
         * turn the picture itself: pl_frame.rotation is a transpose in the
         * output stage, which costs nothing and is the one place the clip's
         * display matrix reaches. The destination rectangle is already the
         * rotated picture's shape, which is what libplacebo asks for. */
        view_zoom_rects(st, &src, &dst);
        view_rot_to_src(st, &src, &image.crop);
        image.rotation = st->dec.rotation;
        sig = view_hash(sig, &src, sizeof(src));
        sig = view_hash(sig, &dst, sizeof(dst));
        sig = view_hash(sig, &image.rotation, sizeof(image.rotation));
    }

    sig = view_hash(sig, &g->pic_serial, sizeof(g->pic_serial));
    sig = view_hash(sig, &st->have_frame, sizeof(st->have_frame));
    sig = view_hash(sig, &ceiling, sizeof(ceiling));
    sig = view_hash(sig, trepr, sizeof(*trepr));
    sig = view_hash(sig, tcolor, sizeof(*tcolor));
    sig = view_hash(sig, &tw, sizeof(tw));
    sig = view_hash(sig, &th, sizeof(th));

    /* The picture is kept and reused until something about it moves. The
     * measured peak is part of that: libplacebo's detector converges over a
     * few passes after a cut, and a paused window has to keep rendering
     * until it has, or the picture would be frozen part-way through the
     * tone mapper settling. The threshold is what makes that terminate. */
    if (g->pic_valid && g->pic_sig == sig &&
        fabs(g->measured_peak - g->pic_peak) <=
        0.002 * CRAW_MAX(g->pic_peak, 1.0))
        return false;

    /* The swapchain's own colour, but none of its bit depth: dithering
     * belongs in the compose, which is the pass that meets the display. */
    pic = (struct pl_frame) {
        .num_planes = 1,
        .planes[0] = {
            .texture           = g->pic_tex,
            .components        = 4,
            .component_mapping = { 0, 1, 2, 3 },
        },
        .repr  = *trepr,
        .color = *tcolor,
        .crop  = dst,
    };
    pic.repr.bits = (struct pl_bit_encoding) { 0 };

    if (!pl_render_image(g->rr, st->have_frame ? &image : NULL, &pic, &rp)) {
        fprintf(stderr, "pl_render_image failed\n");
        g->pic_valid = false;
        return false;
    }

    /* Whatever the detector settled on for the frame just rendered - the
     * window's own, which libplacebo runs by default and nothing here turns
     * off any more. It is reported in PQ, so it comes back through the same
     * rescale libplacebo uses internally. */
    if (pl_renderer_get_hdr_metadata(g->rr, &measured))
        g->measured_peak = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, measured.max_pq_y);
    else
        g->measured_peak = 0.0;

    g->pic_sig = sig;
    g->pic_peak = g->measured_peak;
    g->pic_valid = true;
    g->pic_redrew = true;
    /* Charged to whichever iteration ran the pass, which is normally the
     * one that decoded the picture and not the one that presents it. */
    if (g->rec)
        g->rec->rendered = true;

    return true;
}

/* Runs the picture pass early, off the present path, against the
 * description the last swapchain frame carried. Does nothing before the
 * first present, when there is no such description yet. */
void view_gpu_prepare(struct view_gpu *g, struct view_state *st)
{
    if (!g->sw || !g->pic_have_target)
        return;

    view_gpu_pic_pass(g, st, &g->pic_repr, &g->pic_color, g->pic_tw, g->pic_th);
}

int view_gpu_display(struct view_gpu *g, struct view_state *st)
{
    int64_t t0 = av_gettime_relative();
    if (g->rec)
        g->rec->t_disp = t0;
    struct pl_swapchain_frame sw_frame;
    struct pl_frame target;
    struct pl_overlay overlay;
    struct pl_overlay_part part;
    int tw, th;

    g->pic_redrew = false;

    if (!pl_swapchain_start_frame(g->sw, &sw_frame))
        return CRAW_ERROR(EAGAIN);
    if (g->rec)
        g->rec->t_start = av_gettime_relative();

    pl_frame_from_swapchain(&target, &sw_frame);
    tw = sw_frame.fbo->params.w;
    th = sw_frame.fbo->params.h;

    /* Remembered for view_gpu_prepare(), which has no swapchain frame of
     * its own to ask. Only a resize or a format change moves any of it, and
     * both are in the picture pass's signature. */
    g->pic_repr = target.repr;
    g->pic_color = target.color;
    g->pic_tw = tw;
    g->pic_th = th;
    g->pic_have_target = true;

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

    /* Normally a no-op: view_gpu_prepare() has already put this picture in
     * the texture, off the present path. It still has to be asked, because
     * the first frame, a resize and a look the panel changed a moment ago
     * all reach here with nothing rendered. */
    view_gpu_pic_pass(g, st, &target.repr, &target.color, tw, th);

    if (g->pic_tex && g->pic_valid) {
        /* The compose: source and target are the same size and carry the
         * same colour, so libplacebo has nothing to do here but sample one
         * texel per pixel, blend the overlay and dither into the
         * swapchain's depth. */
        struct pl_render_params crp = pl_render_default_params;
        struct pl_frame src_frame = {
            .num_planes = 1,
            .planes[0] = {
                .texture           = g->pic_tex,
                .components        = 4,
                .component_mapping = { 0, 1, 2, 3 },
            },
            .repr  = target.repr,
            .color = target.color,
            .crop  = { 0, 0, (float)tw, (float)th },
        };

        src_frame.repr.bits = (struct pl_bit_encoding) { 0 };
        crp.peak_detect_params = NULL;
        crp.color_adjustment = &pl_color_adjustment_neutral;
        crp.upscaler = crp.downscaler = NULL;
        crp.skip_anti_aliasing = true;
        crp.info_callback = st->trace.on ? view_trace_pass : NULL;
        target.crop = (pl_rect2df) { 0, 0, (float)tw, (float)th };

        if (!pl_render_image(g->rr_compose, &src_frame, &target, &crp)) {
            fprintf(stderr, "composing the window failed\n");
            return CRAW_ERROR(EIO);
        }
    } else {
        /* No texture to keep the picture in - no floating point format, or
         * no memory. Straight to the swapchain, the way this always
         * worked. */
        struct pl_render_params rp;
        struct pl_color_map_params cmp;
        struct pl_peak_detect_params pdp;
        struct pl_color_adjustment adj;
        struct pl_hdr_metadata measured;
        const struct pl_hook *hook[VIEW_MAX_HOOKS];
        struct pl_frame image;

        view_render_params(st, &rp, &cmp, &pdp, &adj, hook, false, NULL);
        if (st->trace.on)
            rp.info_callback = view_trace_pass;

        if (st->have_frame) {
            pl_rect2df src, dst;

            if (st->dec.raw)
                view_gpu_image(g, &image, view_peak_ceiling(st));
            else
                image = g->mapped;
            view_zoom_rects(st, &src, &dst);
            /* The same pair as the picture pass, and the same turn: `src`
             * is in the rotated picture's pixels and libplacebo wants the
             * source's. */
            view_rot_to_src(st, &src, &image.crop);
            image.rotation = st->dec.rotation;
            target.crop = dst;
        }

        if (!pl_render_image(g->rr, st->have_frame ? &image : NULL,
                             &target, &rp)) {
            fprintf(stderr, "pl_render_image failed\n");
            return CRAW_ERROR(EIO);
        }
        g->pic_redrew = true;
        if (g->rec)
            g->rec->rendered = st->have_frame;

        if (pl_renderer_get_hdr_metadata(g->rr, &measured))
            g->measured_peak = pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS,
                                              measured.max_pq_y);
        else
            g->measured_peak = 0.0;
    }
    if (g->rec)
        g->rec->t_image = av_gettime_relative();
    if (g->rec)
        g->rec->t_hdr = av_gettime_relative();

    if (!pl_swapchain_submit_frame(g->sw))
        return CRAW_ERROR(EIO);

    st->timing.render += (double)(av_gettime_relative() - t0) / 1000.0;
    t0 = av_gettime_relative();
    if (g->rec)
        g->rec->t_render = t0;

    pl_swapchain_swap_buffers(g->sw);

    st->timing.present += (double)(av_gettime_relative() - t0) / 1000.0;
    if (g->rec)
        g->rec->t_present = av_gettime_relative();

    return 0;
}

/* The video export's render. Everything about the source - the renderer,
 * the hooks, the peak detection track, the crop rectangle - is the still
 * export's, so that a frame of a movie and the same frame written on its
 * own are the same picture; everything about the destination belongs to the
 * caller, which is what lets encode.c render straight into an encoder's own
 * Vulkan image or into a texture it reads back. */
int view_gpu_render_target(struct view_gpu *g, struct view_state *st,
                           struct pl_frame *target)
{
    struct pl_render_params rp;
    struct pl_color_map_params cmp;
    struct pl_peak_detect_params pdp;
    struct pl_color_adjustment adj;
    const struct pl_hook *hook[VIEW_MAX_HOOKS];
    struct pl_frame image;
    int cx, cy, cw, ch;

    view_crop_rect(st, &cx, &cy, &cw, &ch);

    if (st->dec.raw)
        view_gpu_image(g, &image, view_peak_ceiling(st));
    else
        image = g->mapped;
    view_render_params(st, &rp, &cmp, &pdp, &adj, hook, true, NULL);

    image.crop = (pl_rect2df) {
        (float)cx, (float)cy, (float)(cx + cw), (float)(cy + ch),
    };
    /* Written upright, the way ffmpeg's autorotate writes it: the target the
     * caller sized already has the rotated shape, and this is what turns the
     * picture into it. The file then carries no display matrix, because its
     * pixels no longer need one. */
    image.rotation = st->dec.rotation;

    if (!pl_render_image(g->rr_export, &image, target, &rp)) {
        fprintf(stderr, "pl_render_image failed while encoding\n");
        return CRAW_ERROR(EIO);
    }

    return 0;
}

int view_gpu_render_export(struct view_gpu *g, struct view_state *st,
                           uint16_t *rgba)
{
    struct pl_render_params rp;
    struct pl_color_map_params cmp;
    struct pl_peak_detect_params pdp;
    struct pl_color_adjustment adj;
    const struct pl_hook *hook[VIEW_MAX_HOOKS];
    struct pl_frame image, target;
    pl_fmt fmt;
    int cx, cy, cw, ch, tw, th;

    /* The target is the crop, or the whole frame when there is none: the
     * export writes source pixels, never a rescale of them. Upright, so its
     * dimensions are the crop's turned - a quarter turn makes a 5760x3040
     * frame 3040x5760 - which is the size ffmpeg's autorotate writes too. */
    view_crop_rect(st, &cx, &cy, &cw, &ch);
    tw = st->dec.rotation % PL_ROTATION_180 ? ch : cw;
    th = st->dec.rotation % PL_ROTATION_180 ? cw : ch;

    if (!g->export_tex || g->export_w != tw || g->export_h != th) {
        pl_tex_destroy(g->gpu, &g->export_tex);

        fmt = pl_find_named_fmt(g->gpu, "rgba16");
        if (!fmt || !(fmt->caps & PL_FMT_CAP_RENDERABLE) ||
            !(fmt->caps & PL_FMT_CAP_HOST_READABLE)) {
            fprintf(stderr, "the GPU cannot render to a readable RGBA16 texture\n");
            return CRAW_ERROR(ENOTSUP);
        }

        g->export_tex = pl_tex_create(g->gpu, pl_tex_params(
            .w             = tw,
            .h             = th,
            .format        = fmt,
            .renderable    = true,
            .storable      = !!(fmt->caps & PL_FMT_CAP_STORABLE),
            .blit_dst      = !!(fmt->caps & PL_FMT_CAP_BLITTABLE),
            .host_readable = true,
        ));
        if (!g->export_tex) {
            fprintf(stderr, "cannot create the %dx%d export texture\n", tw, th);
            return CRAW_ERROR(ENOMEM);
        }
        g->export_w = tw;
        g->export_h = th;
    }

    if (st->dec.raw)
        view_gpu_image(g, &image, view_peak_ceiling(st));
    else
        image = g->mapped;
    view_render_params(st, &rp, &cmp, &pdp, &adj, hook, true, NULL);

    /* The crop is nothing but this rectangle, the same way zoom and pan are
     * nothing but the pair of them on the display path; the turn is
     * libplacebo's, so the file holds the picture the right way up. */
    image.crop = (pl_rect2df) {
        (float)cx, (float)cy, (float)(cx + cw), (float)(cy + ch),
    };
    image.rotation = st->dec.rotation;

    /* Display referred on purpose: what the file holds is the look on
     * screen, sRGB encoded, not the scene-linear numbers. */
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
        .crop  = { 0, 0, (float)tw, (float)th },
    };

    /* The export has its own renderer so that the peak detection track it
     * walks through depends only on the exported frames: the same frame
     * exported twice then gives the same file. */
    if (!pl_render_image(g->rr_export, &image, &target, &rp)) {
        fprintf(stderr, "pl_render_image failed while exporting\n");
        return CRAW_ERROR(EIO);
    }

    /* What the export's own detector measured, beside what the window's
     * shows: the one number that says whether the file was mapped against
     * the same peak as the frame on screen. */
    if (g->verbose) {
        struct pl_hdr_metadata measured;

        if (pl_renderer_get_hdr_metadata(g->rr_export, &measured))
            fprintf(stderr, "export: tone map %s, measured peak %.0f nits, "
                            "window %.0f\n", cmp.tone_mapping_function->name,
                    pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, measured.max_pq_y),
                    g->measured_peak);
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

/* ------------------------------------------------ the direct export path */

/* A scene-referred export never goes near the renderer. The picture is put
 * through the pipeline a second time with CRAWFrameParams.out_space set to
 * the space asked for, into an RGBA32F texture, and the crop of that texture
 * is read straight back: the only libplacebo call these pixels see is
 * pl_tex_download, which is a copy and nothing else. Everything the display
 * path does - the scRGB scale, tone mapping, gamut mapping, the LUT, the
 * grade, the grain - is display-referred by definition and has no business
 * touching linear light on its way to an archive.
 *
 * It costs a second pipeline because the first one writes RGBA16F: libcraw
 * binds the destination as a storage image of the pipeline's own out_format,
 * so a 32 bit target needs a pipeline built for 32 bits. Making the display
 * one RGBA32F instead would double the bandwidth of every frame played for
 * the sake of the frames that are exported, which is the wrong way round. */

void view_gpu_direct_close(struct view_gpu *g)
{
    if (!g->vk)
        return;

    if (g->direct_pipe || g->direct_tex)
        vkDeviceWaitIdle(g->vk->device);

    craw_pipeline_free(&g->direct_pipe);

    if (g->direct_view != VK_NULL_HANDLE) {
        vkDestroyImageView(g->vk->device, g->direct_view, NULL);
        g->direct_view = VK_NULL_HANDLE;
    }
    pl_tex_destroy(g->gpu, &g->direct_tex);
    g->direct_image = VK_NULL_HANDLE;

    if (g->direct_fence != VK_NULL_HANDLE) {
        vkDestroyFence(g->vk->device, g->direct_fence, NULL);
        g->direct_fence = VK_NULL_HANDLE;
    }
    if (g->direct_cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(g->vk->device, g->pool, 1, &g->direct_cmd);
        g->direct_cmd = VK_NULL_HANDLE;
    }

    g->direct_w = g->direct_h = 0;
}

int view_gpu_direct_open(struct view_gpu *g)
{
    CRAWPipelineParams pp;
    VkImageUsageFlags usage;
    VkFormat fmt;
    pl_fmt plfmt;
    int ret;

    if (!g->pipe || !g->width || !g->height) {
        fprintf(stderr, "a scene-referred export needs the raw pipeline, and "
                        "this clip is already developed\n");
        return CRAW_ERROR(ENOTSUP);
    }

    if (g->direct_pipe && g->direct_w == g->width && g->direct_h == g->height &&
        g->direct_debayer == g->pipe_debayer)
        return 0;

    view_gpu_direct_close(g);

    plfmt = pl_find_named_fmt(g->gpu, "rgba32f");
    if (!plfmt || !(plfmt->caps & PL_FMT_CAP_STORABLE) ||
        !(plfmt->caps & PL_FMT_CAP_HOST_READABLE)) {
        fprintf(stderr, "the GPU has no storable, readable RGBA32F format, "
                        "which a scene-referred export is written through\n");
        return CRAW_ERROR(ENOTSUP);
    }

    g->direct_tex = pl_tex_create(g->gpu, pl_tex_params(
        .w             = (int)g->width,
        .h             = (int)g->height,
        .format        = plfmt,
        .storable      = true,
        .host_readable = true,
    ));
    if (!g->direct_tex) {
        fprintf(stderr, "cannot create the %ux%u RGBA32F export texture\n",
                g->width, g->height);
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    g->direct_image = pl_vulkan_unwrap(g->gpu, g->direct_tex, &fmt, &usage);
    if (!g->direct_image || fmt != VK_FORMAT_R32G32B32A32_SFLOAT) {
        fprintf(stderr, "the export texture is not an R32G32B32A32_SFLOAT "
                        "image\n");
        ret = CRAW_ERROR(ENOTSUP);
        goto fail;
    }

    VkImageViewCreateInfo vinfo = {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = g->direct_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = fmt,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    if (vkCreateImageView(g->vk->device, &vinfo, NULL, &g->direct_view) != VK_SUCCESS) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    VkCommandBufferAllocateInfo cb_info = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = g->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkFenceCreateInfo fence_info = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };

    if (vkAllocateCommandBuffers(g->vk->device, &cb_info, &g->direct_cmd) != VK_SUCCESS ||
        vkCreateFence(g->vk->device, &fence_info, NULL, &g->direct_fence) != VK_SUCCESS) {
        ret = CRAW_ERROR(ENOMEM);
        goto fail;
    }

    /* The same stages as the display's, so that what is exported is what is
     * on screen up to the output space: a denoise strength, a false colour
     * pass or a lens warp set in the panel has to be in the file too. The
     * histogram is not: the
     * auto white balance reads the display pipeline's own, and a second set
     * of slots would only be another buffer to zero every frame. The
     * temporal denoise keeps its own state here, which starts empty with
     * every export because this pipeline is built for each one and freed
     * after it; the display's pipeline starts again at the same frame
     * (encode.c), so the two average the same frames. */
    craw_pipeline_params_init(&pp);
    pp.width      = g->width;
    pp.height     = g->height;
    pp.cfa        = CRAW_CFA_RGGB;
    pp.debayer    = g->pipe_debayer;
    pp.out_space  = CRAW_CS_ACES_AP1_LINEAR;
    pp.out_format = VK_FORMAT_R32G32B32A32_SFLOAT;
    pp.flags      = CRAW_PIPELINE_FLAG_DENOISE | CRAW_PIPELINE_FLAG_TEMPORAL_DENOISE |
                    CRAW_PIPELINE_FLAG_FALSE_COLOR | CRAW_PIPELINE_FLAG_CHROMA_DENOISE |
                    CRAW_PIPELINE_FLAG_HIGHLIGHT_RECONSTRUCTION | CRAW_PIPELINE_FLAG_DEFRINGE |
                    CRAW_PIPELINE_FLAG_LENS;

    ret = craw_pipeline_create(g->ctx, &pp, &g->direct_pipe);
    if (ret < 0) {
        fprintf(stderr, "the export pipeline (%ux%u, RGBA32F) failed: %d\n",
                g->width, g->height, ret);
        goto fail;
    }

    g->direct_w = g->width;
    g->direct_h = g->height;
    g->direct_debayer = g->pipe_debayer;

    if (g->verbose)
        fprintf(stderr, "export pipeline: built for %ux%u, RGBA32F\n",
                g->width, g->height);

    return 0;

fail:
    view_gpu_direct_close(g);

    return ret;
}

/* One pass of the export's pipeline into its texture, waited for. */
static int view_gpu_direct_record(struct view_gpu *g, const AVFrame *frame,
                                  const CRAWFrameParams *fp)
{
    AVHWFramesContext *fc;
    AVVulkanFramesContext *vkfc;
    AVVkFrame *vkf;
    const VkFormat *vkfmt;
    VkImageView src_view;
    CRAWImage src, dst;
    uint64_t hold;
    int ret;

    if (!g->direct_pipe)
        return CRAW_ERROR(EINVAL);
    if (!frame || !frame->hw_frames_ctx || frame->format != AV_PIX_FMT_VULKAN)
        return CRAW_ERROR(ENOTSUP);

    fc    = (AVHWFramesContext *)frame->hw_frames_ctx->data;
    vkfc  = fc->hwctx;
    vkf   = (AVVkFrame *)frame->data[0];
    vkfmt = av_vkfmt_from_pixfmt(fc->sw_format);

    if (!vkfmt || vkfmt[0] != VK_FORMAT_R16_UNORM)
        return CRAW_ERROR(ENOTSUP);

    src_view = view_src_view(g, vkf->img[0], vkfmt[0]);
    if (src_view == VK_NULL_HANDLE)
        return CRAW_ERROR(ENOMEM);

    src = (CRAWImage) {
        .image  = vkf->img[0],
        .view   = src_view,
        .format = vkfmt[0],
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .width  = g->direct_w,
        .height = g->direct_h,
    };
    dst = (CRAWImage) {
        .image  = g->direct_image,
        .view   = g->direct_view,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .width  = g->direct_w,
        .height = g->direct_h,
    };

    hold = g->out_value + 1;
    if (!pl_vulkan_hold_ex(g->gpu, pl_vulkan_hold_params(
            .tex       = g->direct_tex,
            .layout    = VK_IMAGE_LAYOUT_GENERAL,
            .qf        = g->qf,
            .semaphore = (pl_vulkan_sem) { g->out_sem, hold },
        ))) {
        fprintf(stderr, "cannot hold the export texture\n");
        return CRAW_ERROR(EBUSY);
    }
    g->out_value = hold;

    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer(g->direct_cmd, &begin) != VK_SUCCESS) {
        ret = CRAW_ERROR(EIO);
        goto release;
    }

    vkfc->lock_frame(fc, vkf);

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

    vkCmdPipelineBarrier2(g->direct_cmd, &(VkDependencyInfo) {
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &to_general,
    });

    ret = craw_pipeline_record(g->direct_pipe, g->direct_cmd, &src, &dst, fp);
    if (ret < 0) {
        vkfc->unlock_frame(fc, vkf);
        vkEndCommandBuffer(g->direct_cmd);
        goto release;
    }

    if (vkEndCommandBuffer(g->direct_cmd) != VK_SUCCESS) {
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
        .commandBuffer = g->direct_cmd,
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

    vkResetFences(g->vk->device, 1, &g->direct_fence);

    view_gpu_lock(g);
    ret = vkQueueSubmit2(g->queue, 1, &submit, g->direct_fence) == VK_SUCCESS
        ? 0 : CRAW_ERROR(EIO);
    view_gpu_unlock(g);

    /* The same hand-back contract view_gpu_process() keeps: the value we
     * signalled is what the next user waits on. */
    if (ret >= 0) {
        vkf->sem_value[0] += 1;
        vkf->layout[0] = VK_IMAGE_LAYOUT_GENERAL;
        vkf->access[0] = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        if (vkf->queue_family[0] != VK_QUEUE_FAMILY_IGNORED)
            vkf->queue_family[0] = g->qf;
    }

    vkfc->unlock_frame(fc, vkf);

    if (ret >= 0 &&
        vkWaitForFences(g->vk->device, 1, &g->direct_fence, VK_TRUE,
                        UINT64_MAX) != VK_SUCCESS)
        ret = CRAW_ERROR(EIO);

release:
    pl_vulkan_release_ex(g->gpu, pl_vulkan_release_params(
        .tex       = g->direct_tex,
        .layout    = VK_IMAGE_LAYOUT_GENERAL,
        .qf        = g->qf,
        .semaphore = (pl_vulkan_sem) { g->out_sem, ret >= 0 ? hold + 1 : hold },
    ));
    if (ret >= 0)
        g->out_value = hold + 1;

    return ret;
}

int view_gpu_direct_frame(struct view_gpu *g, struct view_state *st,
                          enum CRAWColorSpace space, int x, int y, int w, int h,
                          float *rgba, double *gpu_ms, double *dl_ms)
{
    CRAWFrameParams fp;
    int64_t t0;
    int ret;

    view_frame_params(st, &fp);
    /* The one thing that differs from the frame on screen: where the colour
     * stage lands. No CRAW_FRAME_FLAG_CLAMP_NEGATIVE - an archive keeps
     * what the camera matrix produced, negatives included - and no
     * NO_GAMUT_COMPRESSION either, so ACEScg carries the same reference
     * gamut compression the hand-off does and the file holds the numbers
     * the screen was shown. */
    fp.out_space     = space;
    fp.out_space_set = true;
    fp.flags        &= ~(uint32_t)CRAW_FRAME_FLAG_CLAMP_NEGATIVE;

    t0 = av_gettime_relative();

    ret = view_gpu_direct_record(g, st->frame, &fp);
    if (ret < 0)
        return ret;

    if (gpu_ms)
        *gpu_ms = (double)(av_gettime_relative() - t0) / 1000.0;

    t0 = av_gettime_relative();

    /* The crop is a rectangle of the read-back and nothing else: no
     * resample, no second pass, and the rows come out tightly packed
     * because the default row pitch is the rectangle's own width. */
    if (!pl_tex_download(g->gpu, pl_tex_transfer_params(
            .tex = g->direct_tex,
            .rc  = { x, y, 0, x + w, y + h, 1 },
            .ptr = rgba,
        ))) {
        fprintf(stderr, "cannot read the export texture back\n");
        return CRAW_ERROR(EIO);
    }

    if (dl_ms)
        *dl_ms = (double)(av_gettime_relative() - t0) / 1000.0;

    return 0;
}

int view_gpu_direct_preroll(struct view_gpu *g, const AVFrame *frame,
                            const CRAWFrameParams *fp)
{
    return view_gpu_direct_record(g, frame, fp);
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
