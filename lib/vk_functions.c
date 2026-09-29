/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <string.h>

#include <craw/utils.h>

#include "vk_runtime.h"

int craw_vk_functions_load(CRAWVkFunctions *fn, VkInstance instance,
                           VkDevice device, PFN_vkGetInstanceProcAddr iproc,
                           PFN_vkGetDeviceProcAddr dproc)
{
    memset(fn, 0, sizeof(*fn));

    if (!iproc)
        return CRAW_ERROR(EINVAL);

    if (!dproc) {
        dproc = (PFN_vkGetDeviceProcAddr)iproc(instance, "vkGetDeviceProcAddr");
        if (!dproc) {
            craw_log(NULL, CRAW_LOG_ERROR,
                     "craw: failed to load vkGetDeviceProcAddr\n");
            return CRAW_ERROR(ENOSYS);
        }
    }
    fn->GetDeviceProcAddr = dproc;

#define LOAD_I(name)                                                          \
    do {                                                                      \
        fn->name = (PFN_vk##name)iproc(instance, "vk" #name);                 \
        if (!fn->name) {                                                      \
            craw_log(NULL, CRAW_LOG_ERROR,                                    \
                     "craw: failed to load vk" #name "\n");                   \
            return CRAW_ERROR(ENOSYS);                                        \
        }                                                                     \
    } while (0)

#define LOAD_D(name)                                                          \
    do {                                                                      \
        fn->name = (PFN_vk##name)dproc(device, "vk" #name);                   \
        if (!fn->name) {                                                      \
            craw_log(NULL, CRAW_LOG_ERROR,                                    \
                     "craw: failed to load vk" #name "\n");                   \
            return CRAW_ERROR(ENOSYS);                                        \
        }                                                                     \
    } while (0)

/* Entry points promoted to core in 1.3/1.4 only resolve under their
 * un-suffixed name if the caller's device was created with a high enough
 * apiVersion, and only under the KHR name if it instead enabled the
 * standalone extension - try the core name first. */
#define LOAD_D_KHR(name)                                                      \
    do {                                                                      \
        fn->name = (PFN_vk##name)dproc(device, "vk" #name);                   \
        if (!fn->name)                                                        \
            fn->name = (PFN_vk##name)dproc(device, "vk" #name "KHR");         \
        if (!fn->name) {                                                      \
            craw_log(NULL, CRAW_LOG_ERROR,                                    \
                     "craw: failed to load vk" #name "/vk" #name "KHR\n");    \
            return CRAW_ERROR(ENOSYS);                                        \
        }                                                                     \
    } while (0)

    LOAD_I(GetPhysicalDeviceProperties2);
    LOAD_I(GetPhysicalDeviceFeatures2);
    LOAD_I(GetPhysicalDeviceMemoryProperties);
    LOAD_I(GetPhysicalDeviceFormatProperties2);
    LOAD_I(EnumerateDeviceExtensionProperties);
    LOAD_I(GetPhysicalDeviceQueueFamilyProperties);

    LOAD_D(CreateBuffer);
    LOAD_D(DestroyBuffer);
    LOAD_D_KHR(GetBufferDeviceAddress);
    LOAD_D(CreateImage);
    LOAD_D(DestroyImage);
    LOAD_D(CreateImageView);
    LOAD_D(DestroyImageView);

    LOAD_D(CreateShaderModule);
    LOAD_D(DestroyShaderModule);
    LOAD_D(CreateDescriptorSetLayout);
    LOAD_D(DestroyDescriptorSetLayout);
    LOAD_D(CreatePipelineLayout);
    LOAD_D(DestroyPipelineLayout);
    LOAD_D(CreateComputePipelines);
    LOAD_D(DestroyPipeline);

    LOAD_D(CmdBindPipeline);
    LOAD_D_KHR(CmdPushDescriptorSet);
    LOAD_D(CmdPushConstants);
    LOAD_D(CmdDispatch);
    LOAD_D_KHR(CmdPipelineBarrier2);
    LOAD_D(CmdCopyBuffer);
    LOAD_D(CmdCopyBufferToImage);
    LOAD_D(CmdCopyImageToBuffer);
    LOAD_D(CmdFillBuffer);
    LOAD_D(CmdClearColorImage);

    LOAD_D(CreateQueryPool);
    LOAD_D(DestroyQueryPool);
    LOAD_D(GetQueryPoolResults);
    LOAD_D(CmdResetQueryPool);
    LOAD_D_KHR(CmdWriteTimestamp2);

#undef LOAD_I
#undef LOAD_D
#undef LOAD_D_KHR

    return 0;
}
