/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <craw/craw.h>
#include <craw/version.h>
#include <craw/vulkan.h>

#include "vk_runtime.h"

uint32_t craw_version_int(void)
{
    return (CRAW_VERSION_MAJOR << 16) | (CRAW_VERSION_MINOR << 8) |
           (CRAW_VERSION_MICRO << 0);
}

void craw_vlog(CRAWContext *ctx, enum CRAWLogLevel level, const char *fmt,
               va_list args)
{
    if (ctx && level > ctx->opts.log_level)
        return;

    if (ctx && ctx->opts.log_cb) {
        ctx->opts.log_cb(ctx->opts.log_opaque, level, fmt, args);
        return;
    }

    /* Without a callback only messages a library may print unasked reach
     * the user; note the enum counts down with severity. */
    if (level <= CRAW_LOG_WARN)
        vfprintf(stderr, fmt, args);
}

void craw_log(void *ctx, enum CRAWLogLevel level, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    craw_vlog(ctx, level, fmt, args);
    va_end(args);
}

const char *craw_vk_ret2str(VkResult res)
{
    switch (res) {
    case VK_SUCCESS:                        return "success";
    case VK_NOT_READY:                      return "not ready";
    case VK_TIMEOUT:                        return "timeout";
    case VK_EVENT_SET:                      return "event set";
    case VK_EVENT_RESET:                    return "event reset";
    case VK_INCOMPLETE:                     return "incomplete";
    case VK_ERROR_OUT_OF_HOST_MEMORY:       return "out of host memory";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "out of device memory";
    case VK_ERROR_INITIALIZATION_FAILED:    return "initialization failed";
    case VK_ERROR_DEVICE_LOST:              return "device lost";
    case VK_ERROR_MEMORY_MAP_FAILED:        return "memory map failed";
    case VK_ERROR_LAYER_NOT_PRESENT:        return "layer not present";
    case VK_ERROR_EXTENSION_NOT_PRESENT:    return "extension not present";
    case VK_ERROR_FEATURE_NOT_PRESENT:      return "feature not present";
    case VK_ERROR_INCOMPATIBLE_DRIVER:      return "incompatible driver";
    case VK_ERROR_TOO_MANY_OBJECTS:         return "too many objects";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "format not supported";
    case VK_ERROR_FRAGMENTED_POOL:          return "fragmented pool";
    case VK_ERROR_UNKNOWN:                  return "unknown error";
    case VK_ERROR_OUT_OF_POOL_MEMORY:       return "out of pool memory";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE:  return "invalid external handle";
    case VK_ERROR_FRAGMENTATION:            return "fragmentation";
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS:
                                            return "invalid opaque capture address";
    case VK_PIPELINE_COMPILE_REQUIRED:      return "pipeline compile required";
    default:                                return "unknown Vulkan result";
    }
}

int craw_vk_ret2err(VkResult res)
{
    switch (res) {
    case VK_SUCCESS:
        return 0;
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
    case VK_ERROR_OUT_OF_POOL_MEMORY:
    case VK_ERROR_FRAGMENTATION:
        return CRAW_ERROR(ENOMEM);
    case VK_ERROR_DEVICE_LOST:
        return CRAW_ERROR(EIO);
    default:
        return CRAW_ERROR(EINVAL);
    }
}

/* Core in Vulkan 1.3, so never actually required; listed because it is
 * what the packed 8-bit dot product fast path is named after. */
static const char *const craw_optional_dev_extensions[] = {
    "VK_KHR_shader_integer_dot_product",
};

void craw_vulkan_get_requirements(CRAWVulkanRequirements *out)
{
    *out = (CRAWVulkanRequirements) {
        .min_api_version = VK_API_VERSION_1_4,

        /* Everything libcraw uses is core in 1.4. */
        .optional_device_extensions = craw_optional_dev_extensions,
        .nb_optional_device_extensions = CRAW_ARRAY_ELEMS(craw_optional_dev_extensions),

        .required_features = {
            .shaderStorageImageWriteWithoutFormat = VK_TRUE,
            .shaderStorageImageReadWithoutFormat  = VK_TRUE,
            .shaderStorageImageExtendedFormats    = VK_TRUE,
        },
        .required_features_11 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .storageBuffer16BitAccess = VK_TRUE,
        },
        .required_features_12 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
            .bufferDeviceAddress = VK_TRUE,
            .scalarBlockLayout   = VK_TRUE,
            .shaderFloat16       = VK_TRUE,
            .timelineSemaphore   = VK_TRUE,
        },
        .required_features_13 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .synchronization2    = VK_TRUE,
            .subgroupSizeControl = VK_TRUE,
            .computeFullSubgroups = VK_TRUE,
            /* Not in the list in <craw/vulkan.h>, but every libcraw shader
             * takes its workgroup size from specialization constants
             * 253/254/255, which glslang emits as OpExecutionModeId
             * LocalSizeId, and that is only legal with maintenance4
             * enabled. Vulkan 1.4 mandates support for it. */
            .maintenance4        = VK_TRUE,
        },
        .required_features_14 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
            .pushDescriptor = VK_TRUE,
        },
    };

    /* The ready-made chain for VkDeviceCreateInfo. */
    out->required_features_11.pNext = &out->required_features_12;
    out->required_features_12.pNext = &out->required_features_13;
    out->required_features_13.pNext = &out->required_features_14;
    out->required_features_14.pNext = NULL;
    out->features2 = (VkPhysicalDeviceFeatures2) {
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext    = &out->required_features_11,
        .features = out->required_features,
    };
}

/* The features a device reports, queried in one call. */
typedef struct CRAWDeviceFeatures {
    VkPhysicalDeviceFeatures2        f2;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDeviceVulkan14Features f14;
} CRAWDeviceFeatures;

static void craw_query_device_features(PFN_vkGetPhysicalDeviceFeatures2 get,
                                       VkPhysicalDevice phys, CRAWDeviceFeatures *d)
{
    *d = (CRAWDeviceFeatures) {
        .f14 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES },
        .f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES },
        .f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES },
        .f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES },
        .f2  = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 },
    };
    d->f13.pNext = &d->f14;
    d->f12.pNext = &d->f13;
    d->f11.pNext = &d->f12;
    d->f2.pNext  = &d->f11;
    get(phys, &d->f2);
}

/* The first required feature the device lacks, or NULL. */
static const char *craw_first_missing_feature(const CRAWDeviceFeatures *d)
{
    CRAWVulkanRequirements req;

    craw_vulkan_get_requirements(&req);

    const struct {
        const char *name;
        VkBool32 want, have;
    } checks[] = {
        { "shaderStorageImageWriteWithoutFormat",
          req.required_features.shaderStorageImageWriteWithoutFormat,
          d->f2.features.shaderStorageImageWriteWithoutFormat },
        { "shaderStorageImageReadWithoutFormat",
          req.required_features.shaderStorageImageReadWithoutFormat,
          d->f2.features.shaderStorageImageReadWithoutFormat },
        { "shaderStorageImageExtendedFormats",
          req.required_features.shaderStorageImageExtendedFormats,
          d->f2.features.shaderStorageImageExtendedFormats },
        { "storageBuffer16BitAccess",
          req.required_features_11.storageBuffer16BitAccess,
          d->f11.storageBuffer16BitAccess },
        { "bufferDeviceAddress",
          req.required_features_12.bufferDeviceAddress, d->f12.bufferDeviceAddress },
        { "scalarBlockLayout",
          req.required_features_12.scalarBlockLayout, d->f12.scalarBlockLayout },
        { "shaderFloat16",
          req.required_features_12.shaderFloat16, d->f12.shaderFloat16 },
        { "timelineSemaphore",
          req.required_features_12.timelineSemaphore, d->f12.timelineSemaphore },
        { "synchronization2",
          req.required_features_13.synchronization2, d->f13.synchronization2 },
        { "subgroupSizeControl",
          req.required_features_13.subgroupSizeControl, d->f13.subgroupSizeControl },
        { "computeFullSubgroups",
          req.required_features_13.computeFullSubgroups, d->f13.computeFullSubgroups },
        { "maintenance4",
          req.required_features_13.maintenance4, d->f13.maintenance4 },
        { "pushDescriptor",
          req.required_features_14.pushDescriptor, d->f14.pushDescriptor },
    };

    for (size_t i = 0; i < CRAW_ARRAY_ELEMS(checks); i++) {
        if (checks[i].want && !checks[i].have)
            return checks[i].name;
    }

    return NULL;
}

int craw_vulkan_check_device(PFN_vkGetInstanceProcAddr get_proc_addr,
                             VkInstance instance, VkPhysicalDevice phys_device,
                             const char **missing)
{
    PFN_vkGetPhysicalDeviceProperties2 get_props;
    PFN_vkGetPhysicalDeviceFeatures2 get_feats;
    VkPhysicalDeviceProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    };
    CRAWDeviceFeatures d;
    const char *name;

    if (missing)
        *missing = NULL;
    if (!get_proc_addr || !instance || !phys_device)
        return CRAW_ERROR(EINVAL);

    get_props = (PFN_vkGetPhysicalDeviceProperties2)
        get_proc_addr(instance, "vkGetPhysicalDeviceProperties2");
    get_feats = (PFN_vkGetPhysicalDeviceFeatures2)
        get_proc_addr(instance, "vkGetPhysicalDeviceFeatures2");
    if (!get_props || !get_feats)
        return CRAW_ERROR(ENOSYS);

    get_props(phys_device, &props);
    if (props.properties.apiVersion < VK_API_VERSION_1_4) {
        if (missing)
            *missing = "apiVersion";
        return CRAW_ERROR(ENOTSUP);
    }

    craw_query_device_features(get_feats, phys_device, &d);
    name = craw_first_missing_feature(&d);
    if (name) {
        if (missing)
            *missing = name;
        return CRAW_ERROR(ENOTSUP);
    }

    return 0;
}

static int craw_query_properties(CRAWContext *ctx)
{
    VkPhysicalDeviceShaderIntegerDotProductProperties dot = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    };
    VkPhysicalDeviceVulkan13Properties p13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
        .pNext = &dot,
    };
    VkPhysicalDeviceVulkan11Properties p11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
        .pNext = &p13,
    };
    VkPhysicalDeviceProperties2 props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &p11,
    };

    ctx->fn.GetPhysicalDeviceProperties2(ctx->phys_device, &props);
    ctx->fn.GetPhysicalDeviceMemoryProperties(ctx->phys_device, &ctx->mem_props);

    /* The chained structs are stack local; only the core block is kept. */
    ctx->props = props;
    ctx->props.pNext = NULL;

    if (props.properties.apiVersion < VK_API_VERSION_1_4) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "Device \"%s\" reports Vulkan %u.%u, libcraw needs 1.4\n",
                 props.properties.deviceName,
                 VK_API_VERSION_MAJOR(props.properties.apiVersion),
                 VK_API_VERSION_MINOR(props.properties.apiVersion));
        return CRAW_ERROR(ENOTSUP);
    }

    ctx->caps.subgroup_size = p11.subgroupSize;
    ctx->caps.min_subgroup_size = p13.minSubgroupSize;
    ctx->caps.max_subgroup_size = p13.maxSubgroupSize;
    ctx->caps.max_push_size = props.properties.limits.maxPushConstantsSize;
    ctx->caps.max_shmem     = props.properties.limits.maxComputeSharedMemorySize;
    ctx->caps.int8_dot      = dot.integerDotProduct4x8BitPackedSignedAccelerated;

    if (ctx->caps.max_push_size < 128) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "Device \"%s\" allows only %u bytes of push constants, "
                 "libcraw needs 128\n",
                 props.properties.deviceName, ctx->caps.max_push_size);
        return CRAW_ERROR(ENOTSUP);
    }

    craw_log(ctx, CRAW_LOG_VERBOSE,
             "Vulkan device \"%s\" (Vulkan %u.%u), subgroup size %u "
             "(%u to %u), %u bytes of shared memory, %u bytes of push "
             "constants\n",
             props.properties.deviceName,
             VK_API_VERSION_MAJOR(props.properties.apiVersion),
             VK_API_VERSION_MINOR(props.properties.apiVersion),
             ctx->caps.subgroup_size, p13.minSubgroupSize, p13.maxSubgroupSize,
             ctx->caps.max_shmem, ctx->caps.max_push_size);

    return 0;
}

static int craw_check_features(CRAWContext *ctx)
{
    CRAWDeviceFeatures d;
    const char *name;

    craw_query_device_features(ctx->fn.GetPhysicalDeviceFeatures2, ctx->phys_device, &d);
    name = craw_first_missing_feature(&d);
    if (name) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "Device is missing the required Vulkan feature \"%s\"\n", name);
        return CRAW_ERROR(ENOTSUP);
    }

    ctx->caps.float16         = d.f12.shaderFloat16;
    ctx->caps.formatless_read = d.f2.features.shaderStorageImageReadWithoutFormat;

    return 0;
}

int craw_context_create(CRAWContext **out_ctx, const CRAWVulkanInitParams *params,
                        const CRAWContextOptions *opts)
{
    int err;
    CRAWContext *ctx;

    if (!out_ctx || !params || !params->instance || !params->phys_device ||
        !params->device || !params->get_proc_addr)
        return CRAW_ERROR(EINVAL);

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return CRAW_ERROR(ENOMEM);

    if (opts)
        ctx->opts = *opts;
    /* A zeroed options struct means "defaults", and FATAL-only logging is
     * not a useful default; see CRAWContextOptions. */
    if (ctx->opts.log_level == 0)
        ctx->opts.log_level = CRAW_LOG_INFO;

    ctx->instance           = params->instance;
    ctx->phys_device        = params->phys_device;
    ctx->device             = params->device;
    ctx->queue_family_index = params->queue_family_index;
    ctx->alloc              = params->alloc;
    ctx->get_proc_addr      = params->get_proc_addr;

    err = craw_vk_functions_load(&ctx->fn, ctx->instance, ctx->device,
                                 params->get_proc_addr,
                                 params->get_device_proc_addr);
    if (err < 0) {
        craw_log(ctx, CRAW_LOG_ERROR, "Failed to load Vulkan entry points\n");
        goto fail;
    }
    ctx->get_device_proc_addr = ctx->fn.GetDeviceProcAddr;

    err = craw_query_properties(ctx);
    if (err < 0)
        goto fail;

    err = craw_check_features(ctx);
    if (err < 0)
        goto fail;

    err = craw_vk_allocator_create(ctx, &ctx->allocator);
    if (err < 0) {
        craw_log(ctx, CRAW_LOG_ERROR, "Failed to create the memory allocator\n");
        goto fail;
    }

    craw_log(ctx, CRAW_LOG_VERBOSE,
             "libcraw %i.%i.%i on \"%s\": subgroup size %u, fp16 %s, "
             "formatless image reads %s, packed 8-bit dot product %s\n",
             CRAW_VERSION_MAJOR, CRAW_VERSION_MINOR, CRAW_VERSION_MICRO,
             ctx->props.properties.deviceName, ctx->caps.subgroup_size,
             ctx->caps.float16 ? "yes" : "no",
             ctx->caps.formatless_read ? "yes" : "no",
             ctx->caps.int8_dot ? "accelerated" : "emulated");

    *out_ctx = ctx;
    return 0;

fail:
    craw_context_free(&ctx);
    return err;
}

void craw_context_free(CRAWContext **ctx_ptr)
{
    CRAWContext *ctx;

    if (!ctx_ptr || !*ctx_ptr)
        return;
    ctx = *ctx_ptr;

    craw_vk_allocator_destroy(&ctx->allocator);

    free(ctx);
    *ctx_ptr = NULL;
}
