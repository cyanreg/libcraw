/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_VK_RUNTIME_H
#define CRAW_VK_RUNTIME_H

/* Internal Vulkan runtime shared by every stage. Nothing here is public.
 * The runtime never creates or submits command buffers; it records into
 * the one it is given and allocates device memory through VMA on the
 * caller's device. See DESIGN.md section "Vulkan runtime". */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#include <craw/craw.h>
#include <craw/vulkan.h>

/* Every entry point the library uses, loaded through the caller's
 * vkGetInstanceProcAddr / vkGetDeviceProcAddr. Names drop the vk prefix.
 * Entry points promoted to core in 1.3/1.4 are loaded under the core name
 * first and the KHR name second. */
typedef struct CRAWVkFunctions {
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

    /* Instance level */
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;

    /* Device level */
    PFN_vkCreateBuffer CreateBuffer;
    PFN_vkDestroyBuffer DestroyBuffer;
    PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress;
    PFN_vkCreateImage CreateImage;
    PFN_vkDestroyImage DestroyImage;
    PFN_vkCreateImageView CreateImageView;
    PFN_vkDestroyImageView DestroyImageView;

    PFN_vkCreateShaderModule CreateShaderModule;
    PFN_vkDestroyShaderModule DestroyShaderModule;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout CreatePipelineLayout;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
    PFN_vkCreateComputePipelines CreateComputePipelines;
    PFN_vkDestroyPipeline DestroyPipeline;

    PFN_vkCmdBindPipeline CmdBindPipeline;
    PFN_vkCmdPushDescriptorSet CmdPushDescriptorSet;   /* or the KHR entry point */
    PFN_vkCmdPushConstants CmdPushConstants;
    PFN_vkCmdDispatch CmdDispatch;
    PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
    PFN_vkCmdCopyBuffer CmdCopyBuffer;
    PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
    PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
    PFN_vkCmdFillBuffer CmdFillBuffer;
    PFN_vkCmdClearColorImage CmdClearColorImage;

    /* Timestamps, for CRAW_PIPELINE_FLAG_PROFILE only */
    PFN_vkCreateQueryPool CreateQueryPool;
    PFN_vkDestroyQueryPool DestroyQueryPool;
    PFN_vkGetQueryPoolResults GetQueryPoolResults;
    PFN_vkCmdResetQueryPool CmdResetQueryPool;
    PFN_vkCmdWriteTimestamp2 CmdWriteTimestamp2;
} CRAWVkFunctions;

int craw_vk_functions_load(CRAWVkFunctions *fn, VkInstance instance, VkDevice device,
                           PFN_vkGetInstanceProcAddr iproc, PFN_vkGetDeviceProcAddr dproc);

/* Memory allocator (AMD VMA behind a C wrapper, implemented in vma/vma.cpp). */
typedef struct CRAWVkAllocator CRAWVkAllocator;
typedef struct CRAWVkAllocation CRAWVkAllocation;

enum CRAWVkMemUsage {
    CRAW_MEM_DEVICE = 0,     /* device local, not host visible */
    CRAW_MEM_HOST_UPLOAD,    /* host visible, persistently mapped, sequential writes */
    CRAW_MEM_HOST_READBACK,  /* host visible + cached, persistently mapped, random reads */
};

struct CRAWContext {
    CRAWContextOptions opts;

    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    uint32_t         queue_family_index;
    const VkAllocationCallbacks *alloc;
    PFN_vkGetInstanceProcAddr get_proc_addr;
    PFN_vkGetDeviceProcAddr   get_device_proc_addr;

    CRAWVkFunctions fn;

    VkPhysicalDeviceProperties2 props;
    VkPhysicalDeviceMemoryProperties mem_props;

    struct {
        uint32_t subgroup_size;      /* the device's default */
        uint32_t min_subgroup_size;  /* VkPhysicalDeviceVulkan13Properties */
        uint32_t max_subgroup_size;
        uint32_t max_push_size;      /* >= 128 */
        uint32_t max_shmem;          /* maxComputeSharedMemorySize */
        bool     float16;            /* shaderFloat16 */
        bool     int8_dot;           /* integerDotProduct4x8BitPackedSignedAccelerated */
        bool     formatless_read;    /* shaderStorageImageReadWithoutFormat */
    } caps;

    CRAWVkAllocator *allocator;
};

/* Logging: routes to opts.log_cb or stderr. */
void craw_vlog(CRAWContext *ctx, enum CRAWLogLevel level, const char *fmt, va_list args);

/* VkResult helpers */
const char *craw_vk_ret2str(VkResult res);
int craw_vk_ret2err(VkResult res);   /* VK_SUCCESS -> 0, else a negative errno */

/* Allocator */
int  craw_vk_allocator_create(CRAWContext *ctx, CRAWVkAllocator **out);
void craw_vk_allocator_destroy(CRAWVkAllocator **alloc);

/* Creates a VkBuffer/VkImage with bound memory. For host usages `mapped`
 * receives a persistent mapping. Buffers created with
 * VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT get a device address through
 * craw_vk_buffer_create() below. */
int  craw_vk_alloc_buffer(CRAWVkAllocator *a, const VkBufferCreateInfo *info,
                          enum CRAWVkMemUsage usage, VkBuffer *buf,
                          CRAWVkAllocation **alloc, void **mapped);
int  craw_vk_alloc_image(CRAWVkAllocator *a, const VkImageCreateInfo *info,
                         VkImage *img, CRAWVkAllocation **alloc);
void craw_vk_free_buffer(CRAWVkAllocator *a, VkBuffer buf, CRAWVkAllocation *alloc);
void craw_vk_free_image(CRAWVkAllocator *a, VkImage img, CRAWVkAllocation *alloc);
int  craw_vk_alloc_flush(CRAWVkAllocator *a, CRAWVkAllocation *alloc,
                         VkDeviceSize offset, VkDeviceSize size);
int  craw_vk_alloc_invalidate(CRAWVkAllocator *a, CRAWVkAllocation *alloc,
                              VkDeviceSize offset, VkDeviceSize size);

/* Buffers */
typedef struct CRAWVkBuffer {
    VkBuffer         buf;
    CRAWVkAllocation *alloc;
    VkDeviceSize     size;
    VkDeviceAddress  addr;     /* 0 unless created with SHADER_DEVICE_ADDRESS usage */
    void            *mapped;   /* NULL for CRAW_MEM_DEVICE */
    enum CRAWVkMemUsage usage;
} CRAWVkBuffer;

int  craw_vk_buffer_create(CRAWContext *ctx, CRAWVkBuffer *b, VkDeviceSize size,
                           VkBufferUsageFlags usage, enum CRAWVkMemUsage mem);
void craw_vk_buffer_free(CRAWContext *ctx, CRAWVkBuffer *b);
int  craw_vk_buffer_flush(CRAWContext *ctx, CRAWVkBuffer *b);       /* after host writes */
int  craw_vk_buffer_invalidate(CRAWContext *ctx, CRAWVkBuffer *b);  /* before host reads */
void craw_vk_buffer_barrier(CRAWContext *ctx, VkCommandBuffer cmd, CRAWVkBuffer *b,
                            VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                            VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

/* Images. Layout, stage and access are tracked per image so that stages
 * can ask for a transition without knowing who wrote last. */
typedef struct CRAWVkImage {
    VkImage       image;
    VkImageView   view;
    VkFormat      format;
    uint32_t      width, height;

    VkImageLayout         layout;
    VkPipelineStageFlags2 stage;    /* last stage that touched the image */
    VkAccessFlags2        access;   /* and how */

    CRAWVkAllocation *alloc;        /* NULL when wrapped */
    bool owned;
} CRAWVkImage;

/* Device local, optimal tiling, 2D, one mip and layer, starts UNDEFINED. */
int  craw_vk_image_create(CRAWContext *ctx, CRAWVkImage *img, uint32_t width,
                          uint32_t height, VkFormat format, VkImageUsageFlags usage);
/* Adopts a caller image for the duration of one recording. Tracking starts
 * at ext->layout with ALL_COMMANDS / MEMORY_READ|WRITE as the prior use. */
void craw_vk_image_wrap(CRAWVkImage *img, const CRAWImage *ext);
void craw_vk_image_free(CRAWContext *ctx, CRAWVkImage *img);   /* no-op for wrapped */

/* Records a VkImageMemoryBarrier2 from the tracked state to the requested
 * one and updates the tracking. Skips the barrier if nothing changes. */
void craw_vk_image_barrier(CRAWContext *ctx, VkCommandBuffer cmd, CRAWVkImage *img,
                           VkImageLayout new_layout, VkPipelineStageFlags2 dst_stage,
                           VkAccessFlags2 dst_access);

/* Compute shaders. Every shader declares its workgroup size through
 * specialization constants 253, 254 and 255 (the FFmpeg convention) so
 * that the host picks it. All descriptors live in set 0, one array
 * element each, and are pushed with vkCmdPushDescriptorSet. */
typedef struct CRAWVkBinding {
    uint32_t binding;
    VkDescriptorType type;
} CRAWVkBinding;

typedef struct CRAWVkShaderDesc {
    const char *name;
    const uint8_t *spirv;
    size_t spirv_len;

    const CRAWVkBinding *bindings;
    uint32_t nb_bindings;

    uint32_t push_size;              /* bytes, at most 128 */
    uint32_t wg[3];                  /* workgroup size */

    /* Additional specialization constants (ids below 253). May be NULL. */
    const VkSpecializationMapEntry *spec_entries;
    uint32_t nb_spec_entries;
    const void *spec_data;
    size_t spec_size;

    uint32_t required_subgroup_size; /* 0 = no requirement */
} CRAWVkShaderDesc;

typedef struct CRAWVkShader {
    VkShaderModule        module;
    VkDescriptorSetLayout dset_layout;
    VkPipelineLayout      layout;
    VkPipeline            pipeline;
    uint32_t              wg[3];
    uint32_t              push_size;
    char                  name[32];
} CRAWVkShader;

int  craw_vk_shader_create(CRAWContext *ctx, CRAWVkShader *sh, const CRAWVkShaderDesc *desc);
void craw_vk_shader_free(CRAWContext *ctx, CRAWVkShader *sh);

/* One pushed descriptor. Fill with the helpers, then hand the array to
 * craw_vk_shader_push_desc(), which fixes up the info pointers. */
typedef struct CRAWVkDescWrite {
    VkWriteDescriptorSet   write;
    VkDescriptorImageInfo  img;
    VkDescriptorBufferInfo buf;
} CRAWVkDescWrite;

void craw_vk_desc_image(CRAWVkDescWrite *w, uint32_t binding, const CRAWVkImage *img);
void craw_vk_desc_buffer(CRAWVkDescWrite *w, uint32_t binding, VkDescriptorType type,
                         const CRAWVkBuffer *buf, VkDeviceSize offset, VkDeviceSize range);

void craw_vk_shader_bind(CRAWContext *ctx, VkCommandBuffer cmd, const CRAWVkShader *sh);
void craw_vk_shader_push_desc(CRAWContext *ctx, VkCommandBuffer cmd, const CRAWVkShader *sh,
                              CRAWVkDescWrite *writes, uint32_t nb_writes);
void craw_vk_shader_push_const(CRAWContext *ctx, VkCommandBuffer cmd, const CRAWVkShader *sh,
                               const void *data, uint32_t size);
/* Dispatches ceil(w / wg[0]) x ceil(h / wg[1]) x ceil(d / wg[2]) groups. */
void craw_vk_shader_dispatch(CRAWContext *ctx, VkCommandBuffer cmd, const CRAWVkShader *sh,
                             uint32_t w, uint32_t h, uint32_t d);

#endif /* CRAW_VK_RUNTIME_H */
