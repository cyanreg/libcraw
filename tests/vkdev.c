/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkdev.h"

/* This helper is the caller libcraw is written for: it owns the instance,
 * the device, the queue and the command pool, and links the loader so it
 * can use the global vk* prototypes. */

static int craw_td_ret2err(VkResult res)
{
    switch (res) {
    case VK_SUCCESS:
        return 0;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return CRAW_ERROR(ENOMEM);
    case VK_ERROR_DEVICE_LOST:
        return CRAW_ERROR(EIO);
    default:
        return CRAW_ERROR(EINVAL);
    }
}

static uint32_t craw_td_format_bpp(VkFormat fmt)
{
    switch (fmt) {
    case VK_FORMAT_R8_UNORM:               return 1;
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SFLOAT:             return 2;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R32_SFLOAT:             return 4;
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SFLOAT:    return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:    return 16;
    default:                               return 0;
    }
}

static int craw_td_mem_type(CRAWTestDevice *td, uint32_t bits,
                            VkMemoryPropertyFlags want, uint32_t *out)
{
    VkPhysicalDeviceMemoryProperties mp;

    vkGetPhysicalDeviceMemoryProperties(td->phys_device, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i)))
            continue;
        if ((mp.memoryTypes[i].propertyFlags & want) == want) {
            *out = i;
            return 0;
        }
    }

    return CRAW_ERROR(ENOTSUP);
}

/* A mapped staging buffer; the only host visible memory this helper makes. */
typedef struct CRAWTdStaging {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void          *mapped;
} CRAWTdStaging;

static void craw_td_staging_free(CRAWTestDevice *td, CRAWTdStaging *s)
{
    if (s->mapped)
        vkUnmapMemory(td->device, s->mem);
    if (s->buf)
        vkDestroyBuffer(td->device, s->buf, NULL);
    if (s->mem)
        vkFreeMemory(td->device, s->mem, NULL);
    memset(s, 0, sizeof(*s));
}

static int craw_td_staging_create(CRAWTestDevice *td, CRAWTdStaging *s,
                                  VkDeviceSize size, VkBufferUsageFlags usage)
{
    VkResult res;
    int err;
    uint32_t type;
    VkMemoryRequirements req;

    memset(s, 0, sizeof(*s));

    VkBufferCreateInfo buf_info = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    res = vkCreateBuffer(td->device, &buf_info, NULL, &s->buf);
    if (res != VK_SUCCESS)
        return craw_td_ret2err(res);

    vkGetBufferMemoryRequirements(td->device, s->buf, &req);

    err = craw_td_mem_type(td, req.memoryTypeBits,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &type);
    if (err < 0)
        goto fail;

    VkMemoryAllocateInfo alloc_info = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = type,
    };
    res = vkAllocateMemory(td->device, &alloc_info, NULL, &s->mem);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    res = vkBindBufferMemory(td->device, s->buf, s->mem, 0);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    res = vkMapMemory(td->device, s->mem, 0, VK_WHOLE_SIZE, 0, &s->mapped);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    return 0;

fail:
    craw_td_staging_free(td, s);
    return err;
}

/*
 * Device and instance
 */

static bool craw_td_have_layer(const char *name)
{
    uint32_t nb = 0;
    VkLayerProperties *props;
    bool found = false;

    if (vkEnumerateInstanceLayerProperties(&nb, NULL) != VK_SUCCESS || !nb)
        return false;

    props = calloc(nb, sizeof(*props));
    if (!props)
        return false;

    if (vkEnumerateInstanceLayerProperties(&nb, props) == VK_SUCCESS) {
        for (uint32_t i = 0; i < nb && !found; i++)
            found = !strcmp(props[i].layerName, name);
    }

    free(props);
    return found;
}

/* Names the first requirement the device cannot meet, or NULL if it can.
 * The list mirrors craw_vulkan_get_requirements(); a device that fails it
 * would only fail again inside vkCreateDevice. What there is nothing to
 * enable for - the subgroup operations, the formats libcraw reads without
 * a format - is libcraw's own check, which craw_context_create() repeats. */
static const char *craw_td_unsupported(VkInstance instance, VkPhysicalDevice pd)
{
    const char *missing = NULL;
    CRAWVulkanRequirements req;
    VkPhysicalDeviceVulkan14Features f14 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &f14,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &f12,
    };
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f11,
    };

    vkGetPhysicalDeviceFeatures2(pd, &f2);
    craw_vulkan_get_requirements(&req);

    const struct {
        const char *name;
        VkBool32 want, have;
    } checks[] = {
        { "shaderStorageImageWriteWithoutFormat",
          req.required_features.shaderStorageImageWriteWithoutFormat,
          f2.features.shaderStorageImageWriteWithoutFormat },
        { "storageBuffer16BitAccess",
          req.required_features_11.storageBuffer16BitAccess,
          f11.storageBuffer16BitAccess },
        { "bufferDeviceAddress",
          req.required_features_12.bufferDeviceAddress, f12.bufferDeviceAddress },
        { "scalarBlockLayout",
          req.required_features_12.scalarBlockLayout, f12.scalarBlockLayout },
        { "shaderFloat16",
          req.required_features_12.shaderFloat16, f12.shaderFloat16 },
        { "timelineSemaphore",
          req.required_features_12.timelineSemaphore, f12.timelineSemaphore },
        { "synchronization2",
          req.required_features_13.synchronization2, f13.synchronization2 },
        { "subgroupSizeControl",
          req.required_features_13.subgroupSizeControl, f13.subgroupSizeControl },
        { "computeFullSubgroups",
          req.required_features_13.computeFullSubgroups, f13.computeFullSubgroups },
        { "maintenance4",
          req.required_features_13.maintenance4, f13.maintenance4 },
        { "pushDescriptor",
          req.required_features_14.pushDescriptor, f14.pushDescriptor },
    };

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(checks); i++)
        if (checks[i].want && !checks[i].have)
            return checks[i].name;

    if (craw_vulkan_check_device(vkGetInstanceProcAddr, instance, pd, &missing) < 0)
        return missing ? missing : "a libcraw requirement";

    return NULL;
}

static int craw_td_pick_device(CRAWTestDevice *td, const char *name_substr,
                               bool verbose)
{
    VkResult res;
    uint32_t nb = 0;
    VkPhysicalDevice *devs;
    VkPhysicalDevice chosen = VK_NULL_HANDLE, first_gpu = VK_NULL_HANDLE;
    VkPhysicalDevice first_cpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props;
    const bool by_name = name_substr && name_substr[0];

    res = vkEnumeratePhysicalDevices(td->instance, &nb, NULL);
    if (res != VK_SUCCESS || !nb)
        return CRAW_ERROR(ENODEV);

    devs = calloc(nb, sizeof(*devs));
    if (!devs)
        return CRAW_ERROR(ENOMEM);

    res = vkEnumeratePhysicalDevices(td->instance, &nb, devs);
    if (res != VK_SUCCESS) {
        free(devs);
        return craw_td_ret2err(res);
    }

    for (uint32_t i = 0; i < nb && !chosen; i++) {
        const char *missing;
        bool named;

        vkGetPhysicalDeviceProperties(devs[i], &props);
        named = by_name && strstr(props.deviceName, name_substr) != NULL;

        if (verbose)
            fprintf(stderr, "craw: device %u: %s (Vulkan %u.%u)\n", i,
                    props.deviceName,
                    VK_API_VERSION_MAJOR(props.apiVersion),
                    VK_API_VERSION_MINOR(props.apiVersion));

        if (by_name && !named)
            continue;

        if (props.apiVersion < VK_API_VERSION_1_4) {
            if (named || verbose)
                fprintf(stderr, "craw: skipping \"%s\": Vulkan %u.%u, need 1.4\n",
                        props.deviceName,
                        VK_API_VERSION_MAJOR(props.apiVersion),
                        VK_API_VERSION_MINOR(props.apiVersion));
            continue;
        }

        missing = craw_td_unsupported(td->instance, devs[i]);
        if (missing) {
            if (named || verbose)
                fprintf(stderr, "craw: skipping \"%s\": no %s\n",
                        props.deviceName, missing);
            continue;
        }

        if (named)
            chosen = devs[i];
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            if (!first_cpu)
                first_cpu = devs[i];
        } else if (!first_gpu) {
            first_gpu = devs[i];
        }
    }

    if (!chosen)
        chosen = first_gpu ? first_gpu : first_cpu;

    free(devs);

    if (!chosen)
        return CRAW_ERROR(ENODEV);

    vkGetPhysicalDeviceProperties(chosen, &props);
    td->phys_device = chosen;
    td->lavapipe = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    snprintf(td->name, sizeof(td->name), "%s", props.deviceName);

    return 0;
}

static int craw_td_pick_queue(CRAWTestDevice *td)
{
    uint32_t nb = 0;
    VkQueueFamilyProperties *qf;
    int found = -1;

    vkGetPhysicalDeviceQueueFamilyProperties(td->phys_device, &nb, NULL);
    if (!nb)
        return CRAW_ERROR(ENODEV);

    qf = calloc(nb, sizeof(*qf));
    if (!qf)
        return CRAW_ERROR(ENOMEM);
    vkGetPhysicalDeviceQueueFamilyProperties(td->phys_device, &nb, qf);

    /* A dedicated compute family is nicer, but the graphics family is a
     * compute family too and every device has one. */
    for (uint32_t i = 0; i < nb; i++) {
        if (!(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) || !qf[i].queueCount)
            continue;
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            found = (int)i;
            break;
        }
        if (found < 0)
            found = (int)i;
    }

    free(qf);

    if (found < 0)
        return CRAW_ERROR(ENODEV);

    td->queue_family = (uint32_t)found;
    return 0;
}

int craw_testdev_open(CRAWTestDevice *td, const char *name_substr, bool verbose)
{
    VkResult res;
    int err;
    const char *env;
    const char *layers[1];
    uint32_t nb_layers = 0;

    memset(td, 0, sizeof(*td));

    env = getenv("CRAW_TEST_DEVICE");
    if (env && env[0])
        name_substr = env;

    env = getenv("CRAW_TEST_VALIDATION");
    if (env && !strcmp(env, "1")) {
        if (craw_td_have_layer("VK_LAYER_KHRONOS_validation")) {
            layers[nb_layers++] = "VK_LAYER_KHRONOS_validation";
            if (verbose)
                fprintf(stderr, "craw: validation layer enabled\n");
        } else {
            fprintf(stderr, "craw: VK_LAYER_KHRONOS_validation not installed\n");
        }
    }

    VkApplicationInfo app_info = {
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "libcraw",
        .applicationVersion = 1,
        .pEngineName        = "libcraw",
        .engineVersion      = 1,
        .apiVersion         = VK_API_VERSION_1_4,
    };
    VkInstanceCreateInfo inst_info = {
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo        = &app_info,
        .enabledLayerCount       = nb_layers,
        .ppEnabledLayerNames     = nb_layers ? layers : NULL,
    };
    res = vkCreateInstance(&inst_info, NULL, &td->instance);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "craw: unable to create a Vulkan instance (%i)\n", res);
        return craw_td_ret2err(res);
    }

    err = craw_td_pick_device(td, name_substr, verbose);
    if (err < 0) {
        if (name_substr && name_substr[0])
            fprintf(stderr, "craw: no Vulkan 1.4 device matching \"%s\"\n",
                    name_substr);
        else
            fprintf(stderr, "craw: no Vulkan 1.4 device available\n");
        goto fail;
    }

    err = craw_td_pick_queue(td);
    if (err < 0) {
        fprintf(stderr, "craw: \"%s\" has no compute queue\n", td->name);
        goto fail;
    }

    /* libcraw states its requirements once; chain them verbatim. */
    CRAWVulkanRequirements req;
    craw_vulkan_get_requirements(&req);

    VkPhysicalDeviceVulkan14Features feat14 = req.required_features_14;
    VkPhysicalDeviceVulkan13Features feat13 = req.required_features_13;
    VkPhysicalDeviceVulkan12Features feat12 = req.required_features_12;
    VkPhysicalDeviceVulkan11Features feat11 = req.required_features_11;
    feat14.pNext = NULL;
    feat13.pNext = &feat14;
    feat12.pNext = &feat13;
    feat11.pNext = &feat12;
    VkPhysicalDeviceFeatures2 feat2 = {
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext    = &feat11,
        .features = req.required_features,
    };

    float prio = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = td->queue_family,
        .queueCount       = 1,
        .pQueuePriorities = &prio,
    };
    VkDeviceCreateInfo dev_info = {
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &feat2,
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &queue_info,
        .enabledExtensionCount   = req.nb_required_device_extensions,
        .ppEnabledExtensionNames = req.required_device_extensions,
    };
    res = vkCreateDevice(td->phys_device, &dev_info, NULL, &td->device);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "craw: unable to create a device on \"%s\" (%i)\n",
                td->name, res);
        /* A device libcraw cannot run on is a device that does not qualify. */
        err = CRAW_ERROR(ENODEV);
        goto fail;
    }

    vkGetDeviceQueue(td->device, td->queue_family, 0, &td->queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = td->queue_family,
    };
    res = vkCreateCommandPool(td->device, &pool_info, NULL, &td->pool);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    td->init = (CRAWVulkanInitParams) {
        .instance             = td->instance,
        .phys_device          = td->phys_device,
        .device               = td->device,
        .get_proc_addr        = vkGetInstanceProcAddr,
        .get_device_proc_addr = vkGetDeviceProcAddr,
        .queue_family_index   = td->queue_family,
    };

    if (verbose)
        fprintf(stderr, "craw: using \"%s\", queue family %u%s\n", td->name,
                td->queue_family, td->lavapipe ? " (software)" : "");

    return 0;

fail:
    craw_testdev_close(td);
    return err;
}

void craw_testdev_close(CRAWTestDevice *td)
{
    if (!td)
        return;

    if (td->device) {
        vkDeviceWaitIdle(td->device);
        if (td->pool)
            vkDestroyCommandPool(td->device, td->pool, NULL);
        vkDestroyDevice(td->device, NULL);
    }
    if (td->instance)
        vkDestroyInstance(td->instance, NULL);

    memset(td, 0, sizeof(*td));
}

/*
 * Submission
 */

int craw_testdev_run(CRAWTestDevice *td, craw_testdev_record_cb cb, void *opaque)
{
    VkResult res;
    int err;
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    VkCommandBufferAllocateInfo alloc_info = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = td->pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    res = vkAllocateCommandBuffers(td->device, &alloc_info, &cmd);
    if (res != VK_SUCCESS)
        return craw_td_ret2err(res);

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    res = vkBeginCommandBuffer(cmd, &begin_info);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    err = cb(cmd, opaque);
    /* The command buffer must be ended even when the callback gave up, or
     * freeing it is undefined. */
    res = vkEndCommandBuffer(cmd);
    if (err < 0)
        goto fail;
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    VkCommandBufferSubmitInfo cmd_info = {
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
    VkSubmitInfo2 submit = {
        .sType                  = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos    = &cmd_info,
    };
    res = vkQueueSubmit2(td->queue, 1, &submit, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    res = vkQueueWaitIdle(td->queue);
    err = craw_td_ret2err(res);

fail:
    vkFreeCommandBuffers(td->device, td->pool, 1, &cmd);
    return err;
}

typedef struct CRAWTdTimed {
    VkQueryPool pool;
    craw_testdev_record_cb cb;
    void *opaque;
} CRAWTdTimed;

static int craw_td_timed_record(VkCommandBuffer cmd, void *opaque)
{
    CRAWTdTimed *t = opaque;
    int err;

    vkCmdResetQueryPool(cmd, t->pool, 0, 2);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, t->pool, 0);

    err = t->cb(cmd, t->opaque);
    if (err < 0)
        return err;

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, t->pool, 1);
    return 0;
}

/* Timestamps are optional per queue family; returns 0 when this one has
 * none, so callers get a run without a time rather than a bogus one. */
static uint32_t craw_td_timestamp_bits(CRAWTestDevice *td)
{
    uint32_t nb = 0, bits = 0;
    VkQueueFamilyProperties *qf;

    vkGetPhysicalDeviceQueueFamilyProperties(td->phys_device, &nb, NULL);
    if (td->queue_family >= nb)
        return 0;

    qf = calloc(nb, sizeof(*qf));
    if (!qf)
        return 0;
    vkGetPhysicalDeviceQueueFamilyProperties(td->phys_device, &nb, qf);
    bits = qf[td->queue_family].timestampValidBits;
    free(qf);

    return bits;
}

int craw_testdev_run_timed(CRAWTestDevice *td, craw_testdev_record_cb cb,
                           void *opaque, double *ms)
{
    VkResult res;
    int err;
    uint64_t ts[2] = { 0, 0 }, mask;
    uint32_t bits = craw_td_timestamp_bits(td);
    VkPhysicalDeviceProperties props;
    CRAWTdTimed timed = { .cb = cb, .opaque = opaque };

    *ms = 0.0;

    if (!bits)
        return craw_testdev_run(td, cb, opaque);
    mask = bits >= 64 ? UINT64_MAX : ((UINT64_C(1) << bits) - 1);

    VkQueryPoolCreateInfo pool_info = {
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
    };
    res = vkCreateQueryPool(td->device, &pool_info, NULL, &timed.pool);
    if (res != VK_SUCCESS)
        return craw_td_ret2err(res);

    err = craw_testdev_run(td, craw_td_timed_record, &timed);
    if (err < 0)
        goto done;

    res = vkGetQueryPoolResults(td->device, timed.pool, 0, 2, sizeof(ts), ts,
                                sizeof(ts[0]),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto done;
    }

    vkGetPhysicalDeviceProperties(td->phys_device, &props);
    *ms = (double)((ts[1] & mask) - (ts[0] & mask)) *
          (double)props.limits.timestampPeriod / 1e6;

done:
    vkDestroyQueryPool(td->device, timed.pool, NULL);
    return err;
}

/*
 * Images
 */

static int craw_td_to_general(VkCommandBuffer cmd, void *opaque)
{
    VkImage image = *(VkImage *)opaque;

    VkImageMemoryBarrier2 bar = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask        = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask       = 0,
        .dstStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT |
                               VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    VkDependencyInfo dep = {
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &bar,
    };

    vkCmdPipelineBarrier2(cmd, &dep);
    return 0;
}

int craw_testdev_image_create(CRAWTestDevice *td, CRAWTestImage *ti, uint32_t width,
                              uint32_t height, VkFormat format)
{
    VkResult res;
    int err;
    uint32_t type;
    VkMemoryRequirements req;

    memset(ti, 0, sizeof(*ti));

    VkImageCreateInfo img_info = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = { width, height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    res = vkCreateImage(td->device, &img_info, NULL, &ti->img.image);
    if (res != VK_SUCCESS)
        return craw_td_ret2err(res);

    vkGetImageMemoryRequirements(td->device, ti->img.image, &req);

    err = craw_td_mem_type(td, req.memoryTypeBits,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &type);
    if (err < 0)
        goto fail;

    VkMemoryAllocateInfo alloc_info = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = req.size,
        .memoryTypeIndex = type,
    };
    res = vkAllocateMemory(td->device, &alloc_info, NULL, &ti->mem);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    res = vkBindImageMemory(td->device, ti->img.image, ti->mem, 0);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    VkImageViewCreateInfo view_info = {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = ti->img.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    res = vkCreateImageView(td->device, &view_info, NULL, &ti->img.view);
    if (res != VK_SUCCESS) {
        err = craw_td_ret2err(res);
        goto fail;
    }

    ti->img.format = format;
    ti->img.width  = width;
    ti->img.height = height;
    ti->img.layout = VK_IMAGE_LAYOUT_GENERAL;

    err = craw_testdev_run(td, craw_td_to_general, &ti->img.image);
    if (err < 0)
        goto fail;

    return 0;

fail:
    craw_testdev_image_free(td, ti);
    return err;
}

void craw_testdev_image_free(CRAWTestDevice *td, CRAWTestImage *ti)
{
    if (!ti)
        return;

    if (ti->img.view)
        vkDestroyImageView(td->device, ti->img.view, NULL);
    if (ti->img.image)
        vkDestroyImage(td->device, ti->img.image, NULL);
    if (ti->mem)
        vkFreeMemory(td->device, ti->mem, NULL);

    memset(ti, 0, sizeof(*ti));
}

typedef struct CRAWTdCopy {
    CRAWTestImage *ti;
    VkBuffer buf;
    bool to_image;
} CRAWTdCopy;

static int craw_td_copy_record(VkCommandBuffer cmd, void *opaque)
{
    CRAWTdCopy *c = opaque;
    const bool to_image = c->to_image;

    VkImageMemoryBarrier2 pre = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask        = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask       = VK_ACCESS_2_MEMORY_READ_BIT |
                               VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask       = to_image ? VK_ACCESS_2_TRANSFER_WRITE_BIT :
                                          VK_ACCESS_2_TRANSFER_READ_BIT,
        /* vkCmdCopy*Image accepts GENERAL, so the image never leaves the
         * layout the caller of this helper was promised. */
        .oldLayout           = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout           = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = c->ti->img.image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    VkDependencyInfo pre_dep = {
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &pre,
    };
    vkCmdPipelineBarrier2(cmd, &pre_dep);

    VkBufferImageCopy region = {
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel   = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { c->ti->img.width, c->ti->img.height, 1 },
    };

    if (to_image)
        vkCmdCopyBufferToImage(cmd, c->buf, c->ti->img.image,
                               VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    else
        vkCmdCopyImageToBuffer(cmd, c->ti->img.image, VK_IMAGE_LAYOUT_GENERAL,
                               c->buf, 1, &region);

    VkImageMemoryBarrier2 post = pre;
    post.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    post.srcAccessMask = to_image ? VK_ACCESS_2_TRANSFER_WRITE_BIT :
                                    VK_ACCESS_2_TRANSFER_READ_BIT;
    post.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    post.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                         VK_ACCESS_2_MEMORY_WRITE_BIT;

    VkBufferMemoryBarrier2 buf_bar = {
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask        = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask       = VK_ACCESS_2_HOST_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = c->buf,
        .offset              = 0,
        .size                = VK_WHOLE_SIZE,
    };
    VkDependencyInfo post_dep = {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = to_image ? 0 : 1,
        .pBufferMemoryBarriers    = to_image ? NULL : &buf_bar,
        .imageMemoryBarrierCount  = 1,
        .pImageMemoryBarriers     = &post,
    };
    vkCmdPipelineBarrier2(cmd, &post_dep);

    return 0;
}

static int craw_td_image_copy(CRAWTestDevice *td, CRAWTestImage *ti, void *data,
                              size_t size, bool to_image)
{
    int err;
    CRAWTdStaging s;
    uint32_t bpp = craw_td_format_bpp(ti->img.format);
    CRAWTdCopy c = { .ti = ti, .to_image = to_image };

    if (!size)
        return CRAW_ERROR(EINVAL);
    if (bpp && size != (size_t)ti->img.width * ti->img.height * bpp)
        return CRAW_ERROR(EINVAL);

    err = craw_td_staging_create(td, &s, size,
                                 to_image ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT :
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (err < 0)
        return err;

    if (to_image)
        memcpy(s.mapped, data, size);

    c.buf = s.buf;
    err = craw_testdev_run(td, craw_td_copy_record, &c);
    if (err < 0)
        goto done;

    if (!to_image)
        memcpy(data, s.mapped, size);

done:
    craw_td_staging_free(td, &s);
    return err;
}

int craw_testdev_image_upload(CRAWTestDevice *td, CRAWTestImage *ti,
                              const void *data, size_t size)
{
    return craw_td_image_copy(td, ti, (void *)data, size, true);
}

int craw_testdev_image_download(CRAWTestDevice *td, CRAWTestImage *ti, void *data,
                                size_t size)
{
    return craw_td_image_copy(td, ti, data, size, false);
}
