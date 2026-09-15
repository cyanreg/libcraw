/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#ifndef CRAW_TEST_VKDEV_H
#define CRAW_TEST_VKDEV_H

/* Test and tool helper: owns a VkInstance, VkDevice, one compute-capable
 * queue and a command pool, created with everything
 * craw_vulkan_get_requirements() asks for. Links libvulkan directly. Not
 * part of the library. */

#include <stdbool.h>
#include <vulkan/vulkan.h>
#include <craw/vulkan.h>

typedef struct CRAWTestDevice {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    VkCommandPool    pool;
    char             name[256];
    bool             lavapipe;   /* software rasterizer: relax tolerances */

    /* Ready to pass to craw_context_create(). */
    CRAWVulkanInitParams init;
} CRAWTestDevice;

/* Opens the first device whose name contains `name_substr` (NULL: first
 * suitable device; the environment variable CRAW_TEST_DEVICE, if set,
 * overrides `name_substr`). Returns CRAW_ERROR(ENODEV) if none qualifies. */
int  craw_testdev_open(CRAWTestDevice *td, const char *name_substr, bool verbose);
void craw_testdev_close(CRAWTestDevice *td);

/* A 2D image with its own memory, usage STORAGE | TRANSFER_SRC |
 * TRANSFER_DST, transitioned to VK_IMAGE_LAYOUT_GENERAL. */
typedef struct CRAWTestImage {
    CRAWImage      img;
    VkDeviceMemory mem;
} CRAWTestImage;

int  craw_testdev_image_create(CRAWTestDevice *td, CRAWTestImage *ti, uint32_t width,
                               uint32_t height, VkFormat format);
void craw_testdev_image_free(CRAWTestDevice *td, CRAWTestImage *ti);

/* Synchronous upload/download of tightly packed pixels through a staging
 * buffer. `size` must equal width * height * bytes per pixel. The image is
 * left in VK_IMAGE_LAYOUT_GENERAL. */
int craw_testdev_image_upload(CRAWTestDevice *td, CRAWTestImage *ti, const void *data, size_t size);
int craw_testdev_image_download(CRAWTestDevice *td, CRAWTestImage *ti, void *data, size_t size);

/* Records `cb` into a one-shot command buffer, submits it and waits. */
typedef int (*craw_testdev_record_cb)(VkCommandBuffer cmd, void *opaque);
int craw_testdev_run(CRAWTestDevice *td, craw_testdev_record_cb cb, void *opaque);

/* Like craw_testdev_run(), but returns the GPU time of the submission in
 * milliseconds through `ms` using timestamp queries. */
int craw_testdev_run_timed(CRAWTestDevice *td, craw_testdev_record_cb cb, void *opaque, double *ms);

#endif /* CRAW_TEST_VKDEV_H */
