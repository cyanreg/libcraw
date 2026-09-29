/*
 * Copyright © 2026, Lynne
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CRAW_VULKAN_H
#define CRAW_VULKAN_H

#include <vulkan/vulkan.h>

#include <craw/craw.h>

/* Vulkan instance/device extensions and features libcraw needs enabled on
 * the VkDevice it's handed. The caller must query this *before* creating
 * its VkDevice (libcraw never creates one itself) and enable everything
 * listed here.
 *
 * libcraw targets Vulkan 1.4. Everything it uses is core in 1.4:
 *   - VkPhysicalDeviceVulkan12Features: bufferDeviceAddress,
 *     scalarBlockLayout, shaderFloat16, timelineSemaphore
 *   - VkPhysicalDeviceVulkan11Features: storageBuffer16BitAccess
 *   - VkPhysicalDeviceVulkan13Features: synchronization2,
 *     subgroupSizeControl, computeFullSubgroups, maintenance4 (the
 *     shaders take their workgroup size from specialization constants,
 *     which SPIR-V expresses as LocalSizeId and Vulkan only accepts with
 *     maintenance4)
 *   - VkPhysicalDeviceVulkan14Features: pushDescriptor
 *   - VkPhysicalDeviceFeatures: shaderStorageImageWriteWithoutFormat,
 *     shaderStorageImageReadWithoutFormat, shaderStorageImageExtendedFormats
 *   - subgroup operations in compute shaders, Vulkan 1.1 core but reported
 *     rather than enabled (VkPhysicalDeviceSubgroupProperties, or
 *     VkPhysicalDeviceVulkan11Properties): basic and shuffle relative.
 *     They are not part of this struct, having nothing to enable;
 *     craw_vulkan_check_device() and craw_context_create() refuse a
 *     device without them.
 * The feature structs below are for documentation and for callers that
 * chain them verbatim into their VkDeviceCreateInfo; libcraw validates the
 * live device itself via vkGetPhysicalDeviceFeatures2 in
 * craw_context_create(). */
typedef struct CRAWVulkanRequirements {
    uint32_t min_api_version;   /* VK_API_VERSION_1_4 */

    const char *const *required_instance_extensions;
    uint32_t nb_required_instance_extensions;

    const char *const *required_device_extensions;
    uint32_t nb_required_device_extensions;

    /* Extensions that unlock optional fast paths when the caller enables
     * them (e.g. VK_KHR_shader_integer_dot_product). Not read by this
     * version. */
    const char *const *optional_device_extensions;
    uint32_t nb_optional_device_extensions;

    VkPhysicalDeviceFeatures         required_features;
    VkPhysicalDeviceVulkan11Features required_features_11;
    VkPhysicalDeviceVulkan12Features required_features_12;
    VkPhysicalDeviceVulkan13Features required_features_13;
    VkPhysicalDeviceVulkan14Features required_features_14;

    /* The same requirements pre-chained for VkDeviceCreateInfo::pNext or
     * for merging into a caller's own chain: features2.features is
     * required_features and features2.pNext runs through the 11, 12, 13
     * and 14 structs above. The chain points into this struct, so use it
     * in place and do not copy the struct by value afterwards. */
    VkPhysicalDeviceFeatures2        features2;
} CRAWVulkanRequirements;

/* Fills `out` with the extensions/features libcraw requires. Safe to call
 * without a CRAWContext. Pointers in `out` are static and never need to be
 * freed. */
CRAW_API void craw_vulkan_get_requirements(CRAWVulkanRequirements *out);

/* Reports whether a physical device can run libcraw, before any VkDevice
 * exists: its API version and every required feature are queried through
 * `get_proc_addr`. Returns 0 when it can, CRAW_ERROR(ENOTSUP) otherwise
 * with `*missing` (may be NULL) pointing at a static string naming the
 * first missing feature or "apiVersion". Use it to pick a device: on a
 * machine with several GPUs the first one is not necessarily suitable. */
CRAW_API int craw_vulkan_check_device(PFN_vkGetInstanceProcAddr get_proc_addr,
                                      VkInstance instance, VkPhysicalDevice phys_device,
                                      const char **missing);

/* Raw Vulkan handles/state libcraw needs to record compute work. The
 * VkDevice must already have been created with everything reported by
 * craw_vulkan_get_requirements() enabled. libcraw loads its own function
 * pointers via `get_proc_addr` / `get_device_proc_addr` (if the latter is
 * NULL, libcraw obtains vkGetDeviceProcAddr through get_proc_addr). */
typedef struct CRAWVulkanInitParams {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;

    PFN_vkGetInstanceProcAddr get_proc_addr;
    PFN_vkGetDeviceProcAddr   get_device_proc_addr; /* may be NULL */

    /* Queue family the recorded command buffers will be submitted on. Must
     * support VK_QUEUE_COMPUTE_BIT. libcraw never submits to it itself; it
     * only needs the index for ownership fields in barriers. */
    uint32_t queue_family_index;

    /* The device extensions the caller actually enabled at device creation
     * (may be NULL/0). libcraw only activates optional fast paths whose
     * extension appears in this list. */
    const char *const *enabled_dev_extensions;
    uint32_t nb_enabled_dev_extensions;

    const VkAllocationCallbacks *alloc;
} CRAWVulkanInitParams;

/* Creates a CRAWContext bound to the given VkDevice. Fails with
 * CRAW_ERROR(ENOTSUP) if a required extension/feature reported by
 * craw_vulkan_get_requirements() is missing. */
CRAW_API int craw_context_create(CRAWContext **ctx,
                                 const CRAWVulkanInitParams *params,
                                 const CRAWContextOptions *opts);

/* A single 2D image libcraw reads from or writes to. `view` must be a
 * VK_IMAGE_VIEW_TYPE_2D storage-compatible (VK_IMAGE_USAGE_STORAGE_BIT)
 * view of `image` covering one mip/layer. The caller is responsible for
 * transitioning the image to `layout` and for making prior writes visible
 * before craw_pipeline_record(); libcraw only inserts the barriers between
 * its own dispatches and leaves the image in `layout` when the recorded
 * work completes. Pass VK_IMAGE_LAYOUT_GENERAL unless you know better. */
typedef struct CRAWImage {
    VkImage       image;
    VkImageView   view;
    VkFormat      format;
    VkImageLayout layout;
    uint32_t      width, height;
} CRAWImage;

#endif /* CRAW_VULKAN_H */
