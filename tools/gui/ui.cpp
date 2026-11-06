/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The panel, laid out by Dear ImGui and drawn by libplacebo.
 *
 * ImGui turns the panel into triangles every frame, and those are rendered
 * with pl_dispatch_vertex() into a libplacebo texture the size of the
 * window, which the renderer composites as a pl_overlay in the same
 * pl_render_image call as the picture. The overlay is a pl_tex like every
 * other: nothing here records a Vulkan command or holds a semaphore, and
 * libplacebo orders the panel's drawing against the last frame's read of
 * the texture by itself. ImGui's own textures - the font atlas, which
 * grows as glyphs are first drawn - are pl_texs too.
 *
 * ImGui is C++, so this file is. It speaks C to the rest of the program
 * through view.h and is written the way the rest is: static functions, C
 * headers and C strings, nothing of C++ that the panel does not need. */

#include <ctype.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include <fontconfig/fontconfig.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_freetype.h"

#include <libplacebo/dispatch.h>
#include <libplacebo/shaders/custom.h>

extern "C" {
#include <libavutil/time.h>

#include "view.h"
#include "grain.h"
#include "grain_ui.h"
}

/* The embedded film simulation collection is optional: without it the row
 * and its menu compile out and the panel says so. */
#ifndef CINEVIEW_HAVE_LUTS
#define CINEVIEW_HAVE_LUTS 0
#endif

#if CINEVIEW_HAVE_LUTS
extern "C" {
#include "luts.h"
}
#endif

#define VIEW_UI_WIDTH  VIEW_PANEL_WIDTH
/* How close the menu's columns may come to the window's edges. */
#define VIEW_UI_MARGIN 8.0f

/* The panel's face is whatever fontconfig calls sans-serif on this machine,
 * at this many pixels for everything that can be clicked, a size smaller
 * for the readouts, and its bold for the titles. ImGui's own 13 px bitmap
 * face is only what the panel falls back to when fontconfig has nothing or
 * the file will not load. */
#define VIEW_FONT_PX       15.5f
#define VIEW_FONT_SMALL_PX 14.0f

/* What share of a row its label takes, the control having the rest. */
#define VIEW_LABEL_FRAC 0.38f

/* The panel's palette: neutral greys a step apart, text at three
 * strengths, and one accent for whatever is on or chosen. The style is
 * built from it, and so is everything the panel draws by hand. */
#define VIEW_RGB(r, g, b) ImVec4((r) / 255.0f, (g) / 255.0f, (b) / 255.0f, 1.0f)

static const ImVec4 view_col_bg            = VIEW_RGB(21, 22, 25);
static const ImVec4 view_col_surface       = VIEW_RGB(29, 31, 35);
static const ImVec4 view_col_surface_hi    = VIEW_RGB(36, 38, 43);
static const ImVec4 view_col_frame         = VIEW_RGB(35, 37, 42);
static const ImVec4 view_col_frame_hi      = VIEW_RGB(43, 46, 52);
static const ImVec4 view_col_frame_act     = VIEW_RGB(50, 54, 61);
static const ImVec4 view_col_button        = VIEW_RGB(41, 44, 50);
static const ImVec4 view_col_button_hi     = VIEW_RGB(51, 55, 62);
static const ImVec4 view_col_button_act    = VIEW_RGB(59, 64, 72);
static const ImVec4 view_col_border        = VIEW_RGB(46, 49, 55);
static const ImVec4 view_col_text          = VIEW_RGB(226, 228, 233);
static const ImVec4 view_col_text_dim      = VIEW_RGB(142, 146, 155);
static const ImVec4 view_col_text_faint    = VIEW_RGB(96, 99, 107);
static const ImVec4 view_col_accent        = VIEW_RGB(92, 150, 255);
static const ImVec4 view_col_accent_hi     = VIEW_RGB(122, 170, 255);
static const ImVec4 view_col_accent_dim    = VIEW_RGB(44, 70, 118);
static const ImVec4 view_col_accent_dim_hi = VIEW_RGB(53, 84, 140);
static const ImVec4 view_col_warn          = VIEW_RGB(240, 178, 98);
static const ImVec4 view_col_danger        = VIEW_RGB(232, 102, 92);
static const ImVec4 view_col_danger_dim    = VIEW_RGB(120, 46, 42);

/* The cascading menu's own metrics. Its rows are tighter than the panel's:
 * a film category can hold twenty-six stocks and the column still has to
 * fit in a modest window. */
#define VIEW_MENU_ROW   24.0f
#define VIEW_MENU_PAD    4.0f
#define VIEW_MENU_XPAD   8.0f
/* How long the pointer has to rest on an entry before its submenu opens,
 * and how long an already open submenu survives the pointer sweeping
 * across its siblings on the way over to it. */
#define VIEW_MENU_OPEN_US  150000
/* How long a film has to be under the pointer before it previews: long
 * enough that the films a hand crosses on its way into an open column
 * (fifty milliseconds each at a stroll) do not each flash their look. */
#define VIEW_MENU_PEEK_US   80000
#define VIEW_MENU_GUARD_US 300000
#define VIEW_MENU_MIN_W 110.0f
#define VIEW_MENU_MAX_W 330.0f
#define VIEW_MENU_ARROW 16.0f

/* The window ImGui's text input and clipboard go through. */
static SDL_Window *view_ui_window;
/* The face's bold, for titles; the face itself when there is none. */
static ImFont *view_font_strong;
static pl_dispatch view_ui_dp;
static struct pl_vertex_attrib view_ui_attribs[3];
/* When the last frame began, which is ImGui's clock. */
static int64_t view_ui_last;
/* Whether SDL has been asked for text events, so only a change reaches it. */
static bool view_ui_text_on;
/* The clipboard text SDL last handed over, which ImGui only borrows. */
static char *view_ui_clip;
/* The frame's vertices and indices, every draw list's end to end, so that
 * each goes up in one buffer. */
static ImVector<ImDrawVert> view_ui_vtx;
static ImVector<ImDrawIdx> view_ui_idx;

/* What the panel cuts a value that will not fit with: the typographic
 * ellipsis once a face that has it is loaded, three dots until then, since
 * the built-in bitmap face has no U+2026. */
static const char *view_ellipsis = "...";

/* ------------------------------------------------------------ lifetime */

/* The face fontconfig names for "sans-serif", the way every desktop
 * application on the machine finds its UI font, so that the panel gets the
 * one the person already reads everything else in: no option, no
 * environment variable, nothing to configure. `path` is the file and
 * `index` the face inside it for a collection; `ellipsis` says whether
 * the face has U+2026, which fontconfig knows from its charset and which
 * decides what long values are cut with. False when fontconfig has
 * nothing to offer, in which case the built-in face stays.
 *
 * `pattern` is "sans-serif", or that with a weight for the titles' face.
 *
 * fontconfig's whole lifetime is inside this call: nothing else in the
 * program uses it, and the atlas reads the file into memory of its own. */
static bool view_font_find(const char *pattern, char *path, size_t n, int *index,
                           bool *ellipsis, char *family, size_t family_n)
{
    FcPattern *pat, *match = NULL;
    FcResult res = FcResultNoMatch;
    FcChar8 *str;
    FcCharSet *cs;
    bool ok = false;

    if (!FcInit())
        return false;

    pat = FcNameParse((const FcChar8 *)pattern);
    if (pat) {
        FcConfigSubstitute(NULL, pat, FcMatchPattern);
        FcDefaultSubstitute(pat);
        match = FcFontMatch(NULL, pat, &res);
    }

    if (match && res == FcResultMatch &&
        FcPatternGetString(match, FC_FILE, 0, &str) == FcResultMatch) {
        snprintf(path, n, "%s", (const char *)str);
        if (FcPatternGetInteger(match, FC_INDEX, 0, index) != FcResultMatch)
            *index = 0;
        *ellipsis = FcPatternGetCharSet(match, FC_CHARSET, 0, &cs) == FcResultMatch &&
                    FcCharSetHasChar(cs, 0x2026);
        if (FcPatternGetString(match, FC_FAMILY, 0, &str) == FcResultMatch)
            snprintf(family, family_n, "%s", (const char *)str);
        else
            family[0] = '\0';
        ok = true;
    }

    if (match)
        FcPatternDestroy(match);
    if (pat)
        FcPatternDestroy(pat);
    FcFini();

    return ok;
}

/* One face out of fontconfig into the atlas, or NULL. ImGui treats a file
 * it cannot open as a programming error, so the file is opened here first
 * and refused with a line instead. */
static ImFont *view_font_add(struct view_state *st, const char *pattern,
                             bool *ellipsis, char *path, size_t n)
{
    ImGuiIO &io = ImGui::GetIO();
    char family[128];
    ImFont *font = NULL;
    int index = 0;
    FILE *f;

    if (!view_font_find(pattern, path, n, &index, ellipsis, family, sizeof(family))) {
        view_log(st, "panel font: fontconfig has nothing for %s\n", pattern);
        return NULL;
    }

    f = fopen(path, "rb");
    if (f) {
        ImFontConfig cfg;

        fclose(f);
        cfg.FontNo = index;
        font = io.Fonts->AddFontFromFileTTF(path, VIEW_FONT_PX, &cfg);
    }
    if (font)
        view_log(st, "panel font: %s (%s) for %s\n", path, family, pattern);
    else
        view_log(st, "panel font: %s would not load\n", path);

    return font;
}

/* The face and its bold, rasterized by FreeType the way the rest of the
 * desktop's text is, or ImGui's own bitmap face when there is none. The
 * atlas is ImGui's dynamic one: glyphs are rasterized the first time they
 * are drawn, at whatever size they are drawn at, so there is no range to
 * choose and a camera's model name in any script the face covers comes
 * out. A collection's inner faces are reached by their index. The bold is
 * asked for as semibold, which fontconfig answers with the nearest weight
 * the family has. */
static void view_font_load(struct view_state *st)
{
    ImGuiIO &io = ImGui::GetIO();
    char path[VIEW_MAX_PATH], strong_path[VIEW_MAX_PATH];
    bool ellipsis = false, unused;
    ImFont *font;

    io.Fonts->SetFontLoader(ImGuiFreeType::GetFontLoader());

    font = view_font_add(st, "sans-serif", &ellipsis, path, sizeof(path));
    if (!font) {
        view_font_strong = io.Fonts->AddFontDefault();
        return;
    }
    view_ellipsis = ellipsis ? "\xe2\x80\xa6" : "...";

    view_font_strong = view_font_add(st, "sans-serif:weight=semibold", &unused,
                                     strong_path, sizeof(strong_path));
    if (!view_font_strong || !strcmp(path, strong_path))
        view_font_strong = font;
}

/* Square everywhere, room around everything, and the palette above. */
static void view_ui_style(void)
{
    ImGuiStyle &s = ImGui::GetStyle();
    ImVec4 *c = s.Colors;
    ImVec4 select = view_col_accent;

    ImGui::StyleColorsDark(&s);

    s.WindowRounding = 0.0f;
    s.ChildRounding = 0.0f;
    s.FrameRounding = 0.0f;
    s.PopupRounding = 0.0f;
    s.ScrollbarRounding = 0.0f;
    s.GrabRounding = 0.0f;
    s.TabRounding = 0.0f;

    s.WindowPadding = ImVec2(14.0f, 12.0f);
    s.FramePadding = ImVec2(8.0f, 5.0f);
    s.ItemSpacing = ImVec2(8.0f, 6.0f);
    s.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    s.IndentSpacing = 14.0f;
    s.ScrollbarSize = 8.0f;
    s.GrabMinSize = 4.0f;
    s.WindowBorderSize = 0.0f;
    s.PopupBorderSize = 1.0f;
    s.FrameBorderSize = 0.0f;
    s.SeparatorTextBorderSize = 1.0f;
    s.DisabledAlpha = 0.45f;
    s.FontSizeBase = VIEW_FONT_PX;

    select.w = 0.35f;

    c[ImGuiCol_Text]                 = view_col_text;
    c[ImGuiCol_TextDisabled]         = view_col_text_faint;
    c[ImGuiCol_WindowBg]             = view_col_bg;
    c[ImGuiCol_ChildBg]              = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_PopupBg]              = view_col_surface;
    c[ImGuiCol_Border]               = view_col_border;
    c[ImGuiCol_BorderShadow]         = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_FrameBg]              = view_col_frame;
    c[ImGuiCol_FrameBgHovered]       = view_col_frame_hi;
    c[ImGuiCol_FrameBgActive]        = view_col_frame_act;
    c[ImGuiCol_ScrollbarBg]          = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_ScrollbarGrab]        = view_col_frame_act;
    c[ImGuiCol_ScrollbarGrabHovered] = VIEW_RGB(66, 70, 78);
    c[ImGuiCol_ScrollbarGrabActive]  = VIEW_RGB(80, 85, 94);
    c[ImGuiCol_CheckMark]            = view_col_accent;
    c[ImGuiCol_SliderGrab]           = view_col_accent;
    c[ImGuiCol_SliderGrabActive]     = view_col_accent_hi;
    c[ImGuiCol_Button]               = view_col_button;
    c[ImGuiCol_ButtonHovered]        = view_col_button_hi;
    c[ImGuiCol_ButtonActive]         = view_col_button_act;
    c[ImGuiCol_Header]               = view_col_accent_dim;
    c[ImGuiCol_HeaderHovered]        = view_col_accent_dim_hi;
    c[ImGuiCol_HeaderActive]         = view_col_accent_dim_hi;
    c[ImGuiCol_Separator]            = view_col_border;
    c[ImGuiCol_SeparatorHovered]     = view_col_border;
    c[ImGuiCol_SeparatorActive]      = view_col_border;
    c[ImGuiCol_PlotHistogram]        = view_col_accent;
    c[ImGuiCol_PlotHistogramHovered] = view_col_accent_hi;
    c[ImGuiCol_TextSelectedBg]       = select;
}

static const char *view_ui_clip_get(ImGuiContext *)
{
    SDL_free(view_ui_clip);
    view_ui_clip = SDL_GetClipboardText();

    return view_ui_clip;
}

static void view_ui_clip_set(ImGuiContext *, const char *text)
{
    SDL_SetClipboardText(text);
}

int view_ui_open(struct view_state *st, void *sdl_window)
{
    struct view_gpu *g = &st->gpu;
    ImGuiContext *ctx;
    int w = 0, h = 0;
    int ret;

    view_ui_window = (SDL_Window *)sdl_window;
    SDL_GetWindowSizeInPixels(view_ui_window, &w, &h);

    ret = view_gpu_ui_resize(g, w, h);
    if (ret < 0)
        return ret;

    view_ui_dp = pl_dispatch_create(g->gpu->log, g->gpu);
    if (!view_ui_dp)
        return CRAW_ERROR(ENOMEM);

    /* ImDrawVert as it is: two floats of position, two of texture
     * coordinate and four bytes of colour, which the shader reads as
     * normalized. */
    view_ui_attribs[0].name   = "pos";
    view_ui_attribs[0].fmt    = pl_find_vertex_fmt(g->gpu, PL_FMT_FLOAT, 2);
    view_ui_attribs[0].offset = offsetof(ImDrawVert, pos);
    view_ui_attribs[1].name   = "coord";
    view_ui_attribs[1].fmt    = pl_find_vertex_fmt(g->gpu, PL_FMT_FLOAT, 2);
    view_ui_attribs[1].offset = offsetof(ImDrawVert, uv);
    view_ui_attribs[2].name   = "vcolor";
    view_ui_attribs[2].fmt    = pl_find_named_fmt(g->gpu, "rgba8");
    view_ui_attribs[2].offset = offsetof(ImDrawVert, col);
    for (int i = 0; i < 3; i++)
        if (!view_ui_attribs[i].fmt)
            return CRAW_ERROR(ENOTSUP);

    ctx = ImGui::CreateContext();
    if (!ctx)
        return CRAW_ERROR(ENOMEM);

    ImGuiIO &io = ImGui::GetIO();
    ImGuiPlatformIO &pio = ImGui::GetPlatformIO();

    /* Nothing is remembered between runs: the panel's place is fixed, and
     * an imgui.ini dropped into whatever directory the viewer was started
     * from would be litter. */
    io.IniFilename = NULL;
    io.LogFilename = NULL;
    io.BackendPlatformName = "cineview SDL3";
    io.BackendRendererName = "cineview libplacebo";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures |
                       ImGuiBackendFlags_RendererHasVtxOffset;
    /* The pointer is the viewer's: a crosshair while a white pick is armed,
     * the arrow otherwise. */
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    /* A number field is clicked to be typed into, the way nuklear's were,
     * and dragged to be moved. */
    io.ConfigDragClickToInputText = true;
    io.ConfigInputTextCursorBlink = false;
    /* The panel is worked with the pointer: Ctrl+Tab is nobody's window
     * switcher here, and the keys are the transport's. */
    ctx->ConfigNavWindowingKeyNext = ImGuiKey_None;
    ctx->ConfigNavWindowingKeyPrev = ImGuiKey_None;

    pio.Platform_GetClipboardTextFn = view_ui_clip_get;
    pio.Platform_SetClipboardTextFn = view_ui_clip_set;

    view_font_load(st);
    view_ui_style();

    return 0;
}

void view_ui_close(struct view_state *st)
{
    if (ImGui::GetCurrentContext()) {
        for (ImTextureData *t : ImGui::GetPlatformIO().Textures) {
            pl_tex tex = (pl_tex)(uintptr_t)t->TexID;

            if (st->gpu.gpu && tex)
                pl_tex_destroy(st->gpu.gpu, &tex);
            t->SetTexID(ImTextureID_Invalid);
            t->SetStatus(ImTextureStatus_Destroyed);
        }
        ImGui::DestroyContext();
    }

    pl_dispatch_destroy(&view_ui_dp);
    view_ui_vtx.clear();
    view_ui_idx.clear();
    SDL_free(view_ui_clip);
    view_ui_clip = NULL;
    view_ui_window = NULL;
}

bool view_ui_wants_keys(const struct view_state *st)
{
    /* Only a field being typed into takes the keys. The pointer merely
     * hovering the panel must not switch the shortcuts off. */
    return st->ui_edit_active;
}

bool view_ui_locked(const struct view_state *st)
{
    return st->video.active;
}

int view_ui_resize(struct view_state *st, int w, int h)
{
    return view_gpu_ui_resize(&st->gpu, w, h);
}

/* The keys a text field needs, and nothing else: the panel has no keyboard
 * navigation, so every other key is the viewer's. */
static ImGuiKey view_ui_key(SDL_Keycode k)
{
    switch (k) {
    case SDLK_TAB:       return ImGuiKey_Tab;
    case SDLK_LEFT:      return ImGuiKey_LeftArrow;
    case SDLK_RIGHT:     return ImGuiKey_RightArrow;
    case SDLK_UP:        return ImGuiKey_UpArrow;
    case SDLK_DOWN:      return ImGuiKey_DownArrow;
    case SDLK_PAGEUP:    return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN:  return ImGuiKey_PageDown;
    case SDLK_HOME:      return ImGuiKey_Home;
    case SDLK_END:       return ImGuiKey_End;
    case SDLK_INSERT:    return ImGuiKey_Insert;
    case SDLK_DELETE:    return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_SPACE:     return ImGuiKey_Space;
    case SDLK_RETURN:    return ImGuiKey_Enter;
    case SDLK_KP_ENTER:  return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE:    return ImGuiKey_Escape;
    case SDLK_A:         return ImGuiKey_A;
    case SDLK_C:         return ImGuiKey_C;
    case SDLK_V:         return ImGuiKey_V;
    case SDLK_X:         return ImGuiKey_X;
    case SDLK_Y:         return ImGuiKey_Y;
    case SDLK_Z:         return ImGuiKey_Z;
    default:             return ImGuiKey_None;
    }
}

/* ImGui queues what it is handed and trickles it into the frames that
 * follow, one press or release of a button per frame, so a click whose
 * press and release arrive between two frames is still a click: the press
 * is seen by one frame and the release by the next, which is what buttons
 * fire on. view_ui_input_pending() is how the main loop learns that a frame
 * is owed for the half still waiting.
 *
 * The panel is laid out and drawn in drawable pixels, while SDL reports
 * pointer positions in window points; the two differ on a scaled output.
 * Every position is converted here. */
bool view_ui_event(struct view_state *st, void *sdl_event)
{
    const SDL_Event *e = (const SDL_Event *)sdl_event;
    ImGuiIO &io = ImGui::GetIO();
    float sx = st->win_w > 0 ? (float)st->gpu.ui_w / (float)st->win_w : 1.0f;
    float sy = st->win_h > 0 ? (float)st->gpu.ui_h / (float)st->win_h : 1.0f;
    bool down;
    int b;

    switch (e->type) {
    case SDL_EVENT_MOUSE_MOTION:
        io.AddMousePosEvent(e->motion.x * sx, e->motion.y * sy);
        return true;

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
        b = e->button.button == SDL_BUTTON_LEFT   ? ImGuiMouseButton_Left
          : e->button.button == SDL_BUTTON_RIGHT  ? ImGuiMouseButton_Right
          : e->button.button == SDL_BUTTON_MIDDLE ? ImGuiMouseButton_Middle
          : -1;
        if (b < 0)
            return false;
        io.AddMousePosEvent(e->button.x * sx, e->button.y * sy);
        io.AddMouseButtonEvent(b, e->type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        return true;

    case SDL_EVENT_MOUSE_WHEEL:
        io.AddMousePosEvent(e->wheel.mouse_x * sx, e->wheel.mouse_y * sy);
        io.AddMouseWheelEvent(-e->wheel.x, e->wheel.y);
        return true;

    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        ImGuiKey k = view_ui_key(e->key.key);

        down = e->type == SDL_EVENT_KEY_DOWN;
        io.AddKeyEvent(ImGuiMod_Ctrl, (e->key.mod & SDL_KMOD_CTRL) != 0);
        io.AddKeyEvent(ImGuiMod_Shift, (e->key.mod & SDL_KMOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiMod_Alt, (e->key.mod & SDL_KMOD_ALT) != 0);
        io.AddKeyEvent(ImGuiMod_Super, (e->key.mod & SDL_KMOD_GUI) != 0);
        if (k != ImGuiKey_None)
            io.AddKeyEvent(k, down);
        return true;
    }

    /* SDL3 delivers no text at all until a window asks for it, which
     * view_ui_build() does while a field has the keyboard. */
    case SDL_EVENT_TEXT_INPUT:
        io.AddInputCharactersUTF8(e->text.text);
        return true;

    case SDL_EVENT_WINDOW_MOUSE_LEAVE:
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        return true;

    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        io.AddFocusEvent(e->type == SDL_EVENT_WINDOW_FOCUS_GAINED);
        return true;

    default:
        return false;
    }
}

bool view_ui_input_pending(void)
{
    ImGuiContext *g = ImGui::GetCurrentContext();

    return g && g->InputEventsQueue.Size > 0;
}

/* -------------------------------------------------------------- render */

static void view_ui_upload(pl_gpu gpu, pl_tex tex, ImTextureData *t,
                           int x, int y, int w, int h)
{
    struct pl_tex_transfer_params tp = {};

    tp.tex = tex;
    tp.rc.x0 = x;
    tp.rc.y0 = y;
    tp.rc.z0 = 0;
    tp.rc.x1 = x + w;
    tp.rc.y1 = y + h;
    tp.rc.z1 = 1;
    tp.row_pitch = (size_t)t->GetPitch();
    tp.ptr = t->GetPixelsAt(x, y);
    pl_tex_upload(gpu, &tp);
}

/* One of ImGui's textures - in practice the font atlas - created, updated
 * or destroyed as ImGui asks. libplacebo holds a destroyed texture until
 * the frames that sample it are done, so a destroy can be honoured at
 * once. */
static int view_ui_texture(struct view_gpu *g, ImTextureData *t)
{
    pl_tex tex = (pl_tex)(uintptr_t)t->TexID;

    switch (t->Status) {
    case ImTextureStatus_WantDestroy:
        pl_tex_destroy(g->gpu, &tex);
        t->SetTexID(ImTextureID_Invalid);
        t->SetStatus(ImTextureStatus_Destroyed);
        return 0;

    case ImTextureStatus_WantCreate: {
        struct pl_tex_params tp = {};

        if (t->Format != ImTextureFormat_RGBA32)
            return CRAW_ERROR(ENOTSUP);

        tp.w = t->Width;
        tp.h = t->Height;
        tp.format = pl_find_named_fmt(g->gpu, "rgba8");
        tp.sampleable = true;
        tp.host_writable = true;
        tex = tp.format ? pl_tex_create(g->gpu, &tp) : NULL;
        if (!tex)
            return CRAW_ERROR(ENOMEM);

        view_ui_upload(g->gpu, tex, t, 0, 0, t->Width, t->Height);
        t->SetTexID((ImTextureID)(uintptr_t)tex);
        t->SetStatus(ImTextureStatus_OK);
        return 0;
    }

    case ImTextureStatus_WantUpdates:
        for (const ImTextureRect &r : t->Updates)
            view_ui_upload(g->gpu, tex, t, r.x, r.y, r.w, r.h);
        t->SetStatus(ImTextureStatus_OK);
        return 0;

    default:
        return 0;
    }
}

static pl_buf view_ui_buf(pl_gpu gpu, const void *data, size_t size)
{
    struct pl_buf_params bp = {};

    bp.size = size;
    bp.drawable = true;
    bp.initial_data = data;

    return pl_buf_create(gpu, &bp);
}

/* The overlay is cleared and the frame's triangles drawn into it, one
 * dispatch per ImGui draw command: its texture and its clip rectangle,
 * blended the way ImGui expects, which against a cleared target leaves the
 * colour multiplied by its coverage - the premultiplied overlay the
 * renderer composites. The vertices and indices of every draw list go up
 * once, in one buffer each, and are thrown away after: libplacebo frees
 * them once the frame that reads them is done. */
int view_ui_render(struct view_state *st)
{
    static const float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    struct view_gpu *g = &st->gpu;
    ImDrawData *dd = ImGui::GetDrawData();
    int64_t t0 = av_gettime_relative();
    pl_buf vbuf = NULL, ibuf = NULL;
    int vtx_base = 0, idx_base = 0;
    int ret = 0;

    static_assert(sizeof(ImDrawIdx) == 2, "the dispatch below says 16-bit indices");

    if (!g->ui_tex || !dd)
        return 0;

    if (dd->Textures) {
        for (ImTextureData *t : *dd->Textures) {
            if (t->Status == ImTextureStatus_OK)
                continue;
            ret = view_ui_texture(g, t);
            if (ret < 0)
                return ret;
        }
    }

    pl_dispatch_reset_frame(view_ui_dp);
    pl_tex_clear(g->gpu, g->ui_tex, clear);

    if (dd->TotalVtxCount <= 0 || dd->TotalIdxCount <= 0)
        goto done;

    view_ui_vtx.resize(dd->TotalVtxCount);
    view_ui_idx.resize(dd->TotalIdxCount);
    for (const ImDrawList *dl : dd->CmdLists) {
        memcpy(view_ui_vtx.Data + vtx_base, dl->VtxBuffer.Data,
               (size_t)dl->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(view_ui_idx.Data + idx_base, dl->IdxBuffer.Data,
               (size_t)dl->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_base += dl->VtxBuffer.Size;
        idx_base += dl->IdxBuffer.Size;
    }

    vbuf = view_ui_buf(g->gpu, view_ui_vtx.Data,
                       (size_t)view_ui_vtx.Size * sizeof(ImDrawVert));
    ibuf = view_ui_buf(g->gpu, view_ui_idx.Data,
                       (size_t)view_ui_idx.Size * sizeof(ImDrawIdx));
    if (!vbuf || !ibuf) {
        ret = CRAW_ERROR(ENOMEM);
        goto done;
    }

    vtx_base = idx_base = 0;
    for (const ImDrawList *dl : dd->CmdLists) {
        for (const ImDrawCmd &cmd : dl->CmdBuffer) {
            pl_tex tex = (pl_tex)(uintptr_t)cmd.GetTexID();
            struct pl_shader_desc desc = {};
            struct pl_custom_shader cs = {};
            struct pl_dispatch_vertex_params vp = {};
            pl_shader sh;
            float x0 = CRAW_MAX(cmd.ClipRect.x - dd->DisplayPos.x, 0.0f);
            float y0 = CRAW_MAX(cmd.ClipRect.y - dd->DisplayPos.y, 0.0f);
            float x1 = CRAW_MIN(cmd.ClipRect.z - dd->DisplayPos.x, (float)g->ui_w);
            float y1 = CRAW_MIN(cmd.ClipRect.w - dd->DisplayPos.y, (float)g->ui_h);

            /* Nothing of the panel uses a callback. */
            if (cmd.UserCallback || !cmd.ElemCount || !tex ||
                x1 <= x0 || y1 <= y0)
                continue;

            desc.desc.name = "ui_tex";
            desc.desc.type = PL_DESC_SAMPLED_TEX;
            desc.binding.object = tex;
            desc.binding.sample_mode = PL_TEX_SAMPLE_LINEAR;

            cs.description = "ImGui panel";
            cs.body = "color = vcolor * textureLod(ui_tex, coord, 0.0);";
            cs.output = PL_SHADER_SIG_COLOR;
            cs.descriptors = &desc;
            cs.num_descriptors = 1;

            sh = pl_dispatch_begin(view_ui_dp);
            if (!pl_shader_custom(sh, &cs)) {
                pl_dispatch_abort(view_ui_dp, &sh);
                ret = CRAW_ERROR(EINVAL);
                goto done;
            }

            vp.shader = &sh;
            vp.target = g->ui_tex;
            vp.scissors.x0 = (int)x0;
            vp.scissors.y0 = (int)y0;
            vp.scissors.x1 = (int)ceilf(x1);
            vp.scissors.y1 = (int)ceilf(y1);
            vp.blend_params = &pl_alpha_overlay;
            vp.vertex_attribs = view_ui_attribs;
            vp.num_vertex_attribs = 3;
            vp.vertex_stride = sizeof(ImDrawVert);
            vp.vertex_position_idx = 0;
            vp.vertex_coords = PL_COORDS_ABSOLUTE;
            vp.vertex_type = PL_PRIM_TRIANGLE_LIST;
            vp.vertex_count = (int)cmd.ElemCount;
            vp.vertex_buf = vbuf;
            vp.buf_offset = (size_t)(vtx_base + (int)cmd.VtxOffset) * sizeof(ImDrawVert);
            vp.index_fmt = PL_INDEX_UINT16;
            vp.index_buf = ibuf;
            vp.index_offset = (size_t)(idx_base + (int)cmd.IdxOffset) * sizeof(ImDrawIdx);

            if (!pl_dispatch_vertex(view_ui_dp, &vp)) {
                ret = CRAW_ERROR(EINVAL);
                goto done;
            }
        }
        vtx_base += dl->VtxBuffer.Size;
        idx_base += dl->IdxBuffer.Size;
    }

done:
    pl_buf_destroy(g->gpu, &vbuf);
    pl_buf_destroy(g->gpu, &ibuf);
    st->timing.ui += (double)(av_gettime_relative() - t0) / 1000.0;

    return ret;
}

/* -------------------------------------------------------------- panel */

static const char *view_wb_source_name(enum CRAWAwbSource s)
{
    switch (s) {
    case CRAW_AWB_SOURCE_ZONE:   return "surfaces";
    case CRAW_AWB_SOURCE_BRIGHT: return "bright";
    case CRAW_AWB_SOURCE_NONE:   break;
    }
    return "none";
}

static float view_text_w(const char *text)
{
    return ImGui::CalcTextSize(text, NULL, true).x;
}

/* A text cut to fit `w` pixels of the current font, with the ellipsis on
 * the end when it had to be: a value that does not fit its column is shown
 * as far as it goes rather than clipped mid-glyph at the column's edge.
 * The cut steps back over whole UTF-8 sequences, since the values carry a
 * degree sign and a face may put anything in a camera's model name. */
static void view_fit_text(float w, const char *text, char *out, size_t n)
{
    size_t len = strlen(text);
    float ell;

    if (len >= n)
        len = n - 1;

    if (ImGui::CalcTextSize(text, text + len).x <= w) {
        memcpy(out, text, len);
        out[len] = '\0';
        return;
    }

    ell = view_text_w(view_ellipsis);
    while (len > 0) {
        len--;
        /* Not on a continuation byte: back to the start of the sequence. */
        while (len > 0 && ((unsigned char)text[len] & 0xC0) == 0x80)
            len--;
        if (ImGui::CalcTextSize(text, text + len).x + ell <= w)
            break;
    }

    if (len + strlen(view_ellipsis) >= n)
        len = n - 1 - strlen(view_ellipsis);
    memcpy(out, text, len);
    strcpy(out + len, view_ellipsis);
}

/* A text as a printf format that prints exactly it, for the sliders that
 * show a readout in place of their number. */
static void view_fmt_escape(const char *text, char *out, size_t n)
{
    size_t o = 0;

    for (; *text && o + 2 < n; text++) {
        if (*text == '%')
            out[o++] = '%';
        out[o++] = *text;
    }
    out[o] = '\0';
}

/* ------------------------------------------------------- text and rows */

/* A readout: a size smaller and a step dimmer than anything that can be
 * clicked, so that the panel's controls are what the eye finds first and
 * the numbers under them are there when they are looked for. */
static void view_note(const char *fmt, ...) IM_FMTARGS(1);
static void view_note(const char *fmt, ...)
{
    va_list ap;

    ImGui::PushFont(NULL, VIEW_FONT_SMALL_PX);
    ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
    va_start(ap, fmt);
    ImGui::TextV(fmt, ap);
    va_end(ap);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

/* The same, wrapped at the panel's edge: a refusal is a sentence, and a
 * sentence cut at the panel's edge is not one anybody can act on. `col`
 * is the dim text colour, or the warning one for a refusal. */
static void view_note_wrapped(const char *text, const ImVec4 &col)
{
    ImGui::PushFont(NULL, VIEW_FONT_SMALL_PX);
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

/* A row's label, dim, in the column at its left; the cursor is left at
 * the start of the control column with the next widget sized to fill it.
 * Returns the control column's width. */
static float view_prop(const char *label)
{
    float x = ImGui::GetCursorPosX();
    float w = ImGui::GetContentRegionAvail().x;
    float lw = floorf(w * VIEW_LABEL_FRAC);

    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::SameLine(x + lw);
    ImGui::SetNextItemWidth(w - lw);

    return w - lw;
}

/* A label and a value that is only read. */
static void view_prop_value(const char *label, const char *value)
{
    char shown[160];
    float w = view_prop(label);

    view_fit_text(w, value, shown, sizeof(shown));
    ImGui::TextUnformatted(shown);
}

/* Two equal cells across the row, for a control and what it means. */
struct view_row {
    float x, w, sp;
    int n;
};

static view_row view_row_begin(int n)
{
    view_row r;

    r.x = ImGui::GetCursorPosX();
    r.w = ImGui::GetContentRegionAvail().x;
    r.sp = ImGui::GetStyle().ItemSpacing.x;
    r.n = n;

    return r;
}

/* Moves to cell `i` of the row and sizes the next widget to it. Returns
 * the cell's width. */
static float view_cell(const view_row &r, int i)
{
    float cw = (r.w - (float)(r.n - 1) * r.sp) / (float)r.n;

    if (i > 0)
        ImGui::SameLine(r.x + (float)i * (cw + r.sp));
    ImGui::SetNextItemWidth(cw);

    return cw;
}

/* ---------------------------------------------------------- the controls */

static double view_scalar(ImGuiDataType type, const void *p)
{
    switch (type) {
    case ImGuiDataType_S32:   return *(const int *)p;
    case ImGuiDataType_S64:   return (double)*(const int64_t *)p;
    case ImGuiDataType_Float: return *(const float *)p;
    default:                  return 0.0;
    }
}

/* A slider drawn as a bar filled up to its value - from zero, on a range
 * that runs either side of it - with the value written across it. ImGui's
 * own slider does the work: it goes where it is pressed, follows the drag
 * and takes a typed number on Ctrl and a click, held to its ends. Only its
 * drawing is replaced: its frame and its grab are made transparent, and
 * the bar is drawn under them on a channel of its own. True when the value
 * changed. */
static bool view_slider_scalar(const char *id, ImGuiDataType type, void *v,
                               const void *min, const void *max,
                               const char *fmt, ImGuiSliderFlags flags)
{
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const ImVec4 clear(0.0f, 0.0f, 0.0f, 0.0f);
    bool changed;

    dl->ChannelsSplit(2);
    dl->ChannelsSetCurrent(1);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, clear);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, clear);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, clear);
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, clear);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, clear);
    changed = ImGui::SliderScalar(id, type, v, min, max, fmt,
                                  flags | ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopStyleColor(5);

    dl->ChannelsSetCurrent(0);
    if (ImGui::IsItemVisible()) {
        ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        bool active = ImGui::IsItemActive(), hovered = ImGui::IsItemHovered();
        double lo = view_scalar(type, min), hi = view_scalar(type, max);
        double t = hi > lo ? CRAW_CLAMP((view_scalar(type, v) - lo) / (hi - lo), 0.0, 1.0) : 1.0;
        double tz = lo < 0.0 && hi > 0.0 ? -lo / (hi - lo) : 0.0;
        /* Empty at the bottom of the range and full at the top, which an
         * integer slider of a few steps reads as a bar of steps. */
        float xv = a.x + (float)t * (b.x - a.x);
        float x0 = a.x + (float)tz * (b.x - a.x), x1 = xv;
        if (x1 < x0) {
            x1 = x0;
            x0 = xv;
        }

        dl->AddRectFilled(a, b, ImGui::GetColorU32(active  ? view_col_frame_act
                                                  : hovered ? view_col_frame_hi
                                                            : view_col_frame));
        dl->AddRectFilled(ImVec2(x0, a.y), ImVec2(x1, b.y),
                          ImGui::GetColorU32(active || hovered ? view_col_accent_dim_hi
                                                               : view_col_accent_dim));
        /* The bar's moving end in the accent itself, where there is a bar:
         * a value at the origin is said by the number alone. */
        if (x1 - x0 >= 2.0f)
            dl->AddRectFilled(ImVec2(xv == x0 ? x0 : x1 - 2.0f, a.y),
                              ImVec2(xv == x0 ? x0 + 2.0f : x1, b.y),
                              ImGui::GetColorU32(active ? view_col_accent_hi
                                                        : view_col_accent));
    }
    dl->ChannelsMerge();

    return changed;
}

/* A labelled slider row. */
static bool view_slider(const char *label, float *v, float min, float max,
                        const char *fmt)
{
    char id[64];

    snprintf(id, sizeof(id), "##%s", label);
    view_prop(label);
    return view_slider_scalar(id, ImGuiDataType_Float, v, &min, &max, fmt, 0);
}

static bool view_slider_int(const char *label, int *v, int min, int max,
                            const char *fmt)
{
    char id[64];

    snprintf(id, sizeof(id), "##%s", label);
    view_prop(label);
    return view_slider_scalar(id, ImGuiDataType_S32, v, &min, &max, fmt, 0);
}

/* A number field: dragged sideways to move it, clicked to type into it, in
 * whatever width the row gave it. `speed` is how much a pixel of drag is
 * worth. */
static bool view_number(const char *id, int *v, int min, int max, float speed,
                        const char *fmt)
{
    return ImGui::DragInt(id, v, speed, min, max, fmt, ImGuiSliderFlags_AlwaysClamp);
}

/* A labelled number row. */
static bool view_number_row(const char *label, int *v, int min, int max,
                            float speed, const char *fmt)
{
    char id[64];

    snprintf(id, sizeof(id), "##%s", label);
    view_prop(label);
    return view_number(id, v, min, max, speed, fmt);
}

/* A labelled combo row, its list opening under it, with the same dim
 * chevron the menu rows have rather than ImGui's arrow box. */
static bool view_combo_row(const char *label, int *v, const char *const *items,
                           int n, int height = -1)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    float fs = ImGui::GetFontSize();
    bool changed = false;
    char id[64];
    float cw;

    snprintf(id, sizeof(id), "##%s", label);
    cw = view_prop(label);
    ImVec2 a = ImGui::GetCursorScreenPos();

    if (height > 0)
        ImGui::SetNextWindowSizeConstraints(ImVec2(0.0f, 0.0f),
            ImVec2(FLT_MAX, ImGui::GetTextLineHeightWithSpacing() * (float)height +
                            2.0f * style.WindowPadding.y));
    if (ImGui::BeginCombo(id, *v >= 0 && *v < n ? items[*v] : "",
                          ImGuiComboFlags_NoArrowButton)) {
        for (int i = 0; i < n; i++) {
            bool on = i == *v;

            if (ImGui::Selectable(items[i], on) && !on) {
                *v = i;
                changed = true;
            }
            if (on)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::RenderArrow(dl, ImVec2(a.x + cw - style.FramePadding.x - fs * 0.6f,
                                  a.y + style.FramePadding.y + fs * 0.2f),
                       ImGui::GetColorU32(view_col_text_dim), ImGuiDir_Down, 0.6f);

    return changed;
}

/* A choice of a few, as one strip of buttons with the chosen one lit: what
 * a row of radio buttons says, in less room and at a glance. Fills what is
 * left of the row. True when the choice changed. */
static bool view_segmented(const char *id, const char *const *labels, int n,
                           int *sel)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    float w = ImGui::GetContentRegionAvail().x;
    float bw = floorf((w - (float)(n - 1)) / (float)n);
    bool changed = false;

    ImGui::PushID(id);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1.0f, style.ItemSpacing.y));
    for (int i = 0; i < n; i++) {
        bool on = *sel == i;

        if (i)
            ImGui::SameLine();
        if (on) {
            ImGui::PushStyleColor(ImGuiCol_Button, view_col_accent_dim);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, view_col_accent_dim_hi);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, view_col_accent_dim_hi);
        }
        ImGui::PushID(i);
        /* The last takes whatever rounding left over, so the strip ends
         * flush with the row. */
        if (ImGui::Button(labels[i],
                          ImVec2(i == n - 1 ? w - (bw + 1.0f) * (float)(n - 1) : bw, 0.0f)) &&
            !on) {
            *sel = i;
            changed = true;
        }
        ImGui::PopID();
        if (on)
            ImGui::PopStyleColor(3);
    }
    ImGui::PopStyleVar();
    ImGui::PopID();

    return changed;
}

/* A button in the accent, for the one thing a section is there to do. */
static bool view_button_primary(const char *label, const ImVec2 &size)
{
    bool pressed;

    ImGui::PushStyleColor(ImGuiCol_Button, view_col_accent_dim);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, view_col_accent_dim_hi);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, view_col_accent);
    pressed = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);

    return pressed;
}

static bool view_button_danger(const char *label, const ImVec2 &size)
{
    bool pressed;

    ImGui::PushStyleColor(ImGuiCol_Button, view_col_danger_dim);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, view_col_danger);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, view_col_danger);
    pressed = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);

    return pressed;
}

/* A checkbox with three states, which ImGui does not have: empty for 0,
 * and for 1 and 2 the box half filled along its diagonal, the upper left
 * triangle for 1 and the lower right for 2, in the colour a checked box is
 * filled with. Built of ImGui's own pieces - its item, its button
 * behaviour and its frame - so it looks, hovers and greys out as every
 * other checkbox of the panel does. A click moves the state on, 0, 1, 2
 * and round. True when it did. */
static bool view_check3(const char *id, int *state)
{
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    const ImGuiStyle &style = ImGui::GetStyle();
    float sz = ImGui::GetFrameHeight();
    ImGuiID gid;
    bool hovered, held, pressed;

    if (win->SkipItems)
        return false;

    gid = win->GetID(id);
    ImRect bb(win->DC.CursorPos, ImVec2(win->DC.CursorPos.x + sz,
                                        win->DC.CursorPos.y + sz));
    ImGui::ItemSize(bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, gid))
        return false;

    pressed = ImGui::ButtonBehavior(bb, gid, &hovered, &held);
    if (pressed)
        *state = (*state + 1) % 3;

    ImGui::RenderFrame(bb.Min, bb.Max,
                       ImGui::GetColorU32(held && hovered ? ImGuiCol_FrameBgActive
                                          : hovered       ? ImGuiCol_FrameBgHovered
                                                          : ImGuiCol_FrameBg),
                       true, style.FrameRounding);

    if (*state) {
        float pad = CRAW_MAX(1.0f, floorf(sz / 6.0f));
        ImVec2 a(bb.Min.x + pad, bb.Min.y + pad), b(bb.Max.x - pad, bb.Max.y - pad);
        ImU32 col = ImGui::GetColorU32(ImGuiCol_CheckMark);

        if (*state == 1)
            win->DrawList->AddTriangleFilled(a, ImVec2(b.x, a.y), ImVec2(a.x, b.y), col);
        else
            win->DrawList->AddTriangleFilled(ImVec2(b.x, a.y), b, ImVec2(a.x, b.y), col);
    }

    return pressed;
}

/* The three sections a scene-referred export takes out of the picture: the
 * look, the adjustments and the grain. All three are display-referred
 * operators on values in [0,1], and a scene-referred export is written
 * straight out of the pipeline with none of them on it - so they are drawn
 * greyed and take no input. The same hooks are left off the display too,
 * which is the point: what is on screen is what the file holds, up to the
 * display transform, so a live control here would be lying about the file
 * and about the window at once.
 *
 * The section's own header is deliberately left alive, because a section
 * you cannot open is a section that cannot tell you why it is greyed. The
 * one line that explains the rest is drawn before the greying, in the
 * warning colour: it is the one line that has to be read. */
static bool view_grey_begin(struct view_state *st, const char *why)
{
    if (!view_export_is_scene(st))
        return false;

    if (why)
        view_note_wrapped(why, view_col_warn);
    ImGui::BeginDisabled();

    return true;
}

static void view_grey_end(bool on)
{
    if (on)
        ImGui::EndDisabled();
}

/* ------------------------------------------------- the section headers */

/* A section header is a bar across the panel's whole width: the section's
 * power switch at its left, then a chevron and the section's title in the
 * bold face, the way RawTherapee gives every tool an enable button beside
 * its title: off bypasses the whole section while every
 * parameter in it is kept, so switching it back on returns the picture it
 * had. The bar opens and closes the body; the switch only switches.
 *
 * The switch is a dot, filled in the accent when on and an empty ring when
 * off, so that down a column of ten of them the state reads without
 * reading. A greyed one is drawn off whatever the setting is: nothing is
 * being applied, and a lit switch that cannot be clicked read as forced on
 * (Lynne, of Lens on a still with no record). The setting itself is kept
 * for the next clip the section can work on.
 *
 * Which sections are open lives in view state rather than in ImGui's own
 * table, so that it carries from clip to clip like every other setting.
 * Both halves are clicked; nothing on the keyboard reaches them.
 *
 * `on` is the section's switch, or NULL for a section with nothing to
 * bypass. `disabled` greys the switch without changing it, which is what a
 * scene-referred export does to the three display-referred sections.
 * Returns true when the body is to be laid out, and then view_section_end()
 * closes it. */
static bool view_section_header(struct view_state *st, enum view_section id,
                                const char *title, bool *on, bool disabled)
{
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    const ImGuiStyle &style = ImGui::GetStyle();
    bool open = st->sections.open[id];
    float fs = ImGui::GetFontSize();
    float h = ImGui::GetFrameHeight() + 8.0f;
    float px = style.WindowPadding.x;
    float r = floorf(fs * 0.38f);
    bool hovered = false, held = false;

    if (win->SkipItems)
        return false;

    /* From one edge of the panel to the other, through its padding; the
     * row takes exactly its height, so closed sections stack bar on bar. */
    ImVec2 pos = win->DC.CursorPos;
    ImRect row(ImVec2(win->InnerRect.Min.x, pos.y), ImVec2(win->InnerRect.Max.x, pos.y + h));
    /* The switch at the left, then the chevron and the title. A section
     * with nothing to bypass keeps the switch's slot empty, so every
     * title in the column lines up. */
    ImVec2 c(row.Min.x + px + r, row.GetCenter().y);
    ImRect sw(row.Min, ImVec2(c.x + r + 6.0f, row.Max.y));
    ImRect head(ImVec2(on ? sw.Max.x : row.Min.x, row.Min.y), row.Max);
    float tx = sw.Max.x + 4.0f;

    ImGui::PushID((int)id);
    ImGui::ItemSize(ImVec2(0.0f, h - style.ItemSpacing.y));

    ImGuiID hid = win->GetID("##head");
    if (ImGui::ItemAdd(head, hid) &&
        ImGui::ButtonBehavior(head, hid, &hovered, &held))
        open = !open;

    ImGui::PushClipRect(row.Min, row.Max, false);
    win->DrawList->AddRectFilled(row.Min, row.Max,
                                 ImGui::GetColorU32(hovered ? view_col_surface_hi
                                                            : view_col_surface));
    win->DrawList->AddLine(row.Min, ImVec2(row.Max.x, row.Min.y),
                           ImGui::GetColorU32(view_col_border));
    ImGui::RenderArrow(win->DrawList,
                       ImVec2(tx, row.Min.y + (h - fs * 0.6f) * 0.5f),
                       ImGui::GetColorU32(view_col_text_dim),
                       open ? ImGuiDir_Down : ImGuiDir_Right, 0.6f);
    win->DrawList->AddText(view_font_strong, fs,
                           ImVec2(tx + fs * 0.6f + 8.0f, row.Min.y + (h - fs) * 0.5f),
                           ImGui::GetColorU32(ImGuiCol_Text), title);

    if (on) {
        bool sw_hovered = false, sw_held = false, lit;
        ImGuiID sid;

        ImGui::BeginDisabled(disabled);
        sid = win->GetID("##on");
        if (ImGui::ItemAdd(sw, sid) &&
            ImGui::ButtonBehavior(sw, sid, &sw_hovered, &sw_held))
            *on = !*on;

        lit = *on && !disabled;
        ImU32 col = ImGui::GetColorU32(lit ? (sw_hovered ? view_col_accent_hi : view_col_accent)
                                           : (sw_hovered ? view_col_text_dim : view_col_text_faint));

        if (lit)
            win->DrawList->AddCircleFilled(c, r, col);
        else
            win->DrawList->AddCircle(c, r - 0.75f, col, 0, 1.5f);
        ImGui::EndDisabled();
    }

    ImGui::PopClipRect();
    ImGui::PopID();

    st->sections.open[id] = open;
    if (open)
        ImGui::Dummy(ImVec2(0.0f, 2.0f));

    return open;
}

static void view_section_end(void)
{
    ImGui::Dummy(ImVec2(0.0f, 4.0f));
}

/* ------------------------------------------------- the cascading menus */

/* Three rows use this: the film simulation, the video export's codec
 * preset and the export's colour space. The geometry, the hover
 * hysteresis and the drawing are one implementation; what differs is only
 * what a column holds, which the five accessors below answer per kind.
 *
 * ImGui has menus that cascade, but not with the timing a list of a few
 * hundred film renderings needs: an entry rested on before it opens, a
 * guard while the pointer travels to a submenu, the films a hand sweeps
 * across not each previewing their look. So the columns are plain ImGui
 * windows that ImGui only draws, borders and clips - they sit over the
 * panel and the picture and keep the pointer from reaching what is under
 * them - and the hover, click and scroll logic below works straight off
 * the pointer position against rectangles this file computes. */

#if CINEVIEW_HAVE_LUTS
/* The film at entry `i` of the films column, under the category the path
 * has open, or NULL. */
static const ViewLutFilm *view_menu_film_at(const struct view_state *st, int i)
{
    const struct view_lut_menu *m = &st->menu;
    const ViewLutFilmCategory *c;

    if (m->sel[0] != 0 || m->sel[1] < 0 || i < 0)
        return NULL;
    c = view_lut_film_category(m->sel[1]);
    if (!c || i >= c->n_films)
        return NULL;

    return view_lut_film(c->first_film + i);
}

/* The film whose renderings column 3 lists, or NULL when the path does not
 * reach that far. */
static const ViewLutFilm *view_menu_film(const struct view_state *st)
{
    return view_menu_film_at(st, st->menu.sel[2]);
}

/* The rendering that stands for a film while the film itself is what the
 * highlight is on: the one labelled "base", or failing that the first
 * whose label mentions it ("NC, base"), then the first that is neither a
 * push nor a pull, then the first there is. Six films have no base at
 * all - the print stocks are three gamut mappings each, Velvia 100 a
 * single generic profile - and the Polaroids call theirs "cold" and
 * "warm", which the third rule reaches. */
static int view_menu_film_base(const ViewLutFilm *f)
{
    int found = -1;

    if (!f || f->n_luts <= 0)
        return -1;

    for (int i = f->first_lut; i < f->first_lut + f->n_luts; i++) {
        const ViewLut *l = view_lut_get(i);

        if (!l)
            continue;
        if (!strcmp(l->label, "base"))
            return i;
        if (found < 0 && strstr(l->label, "base"))
            found = i;
    }
    if (found >= 0)
        return found;

    for (int i = f->first_lut; i < f->first_lut + f->n_luts; i++) {
        const ViewLut *l = view_lut_get(i);

        if (l && !strstr(l->label, "stop"))
            return i;
    }

    return f->first_lut;
}
#endif

/* The colour spaces are grouped by their first column entry, so that the
 * three BT.2020 transfers hang off one row. */
static int view_menu_colour_groups(void)
{
    int n = 0;

    for (int i = 0; i < VIEW_NB_COLOURS; i++)
        if (!i || strcmp(view_colours[i].group, view_colours[i - 1].group))
            n++;

    return n;
}

/* The first and the count of the colours in group `g`. */
static int view_menu_colour_group(int g, int *count)
{
    int seen = -1;

    for (int i = 0; i < VIEW_NB_COLOURS; i++) {
        if (!i || strcmp(view_colours[i].group, view_colours[i - 1].group))
            seen++;
        if (seen != g)
            continue;

        *count = 0;
        for (int j = i; j < VIEW_NB_COLOURS &&
                        !strcmp(view_colours[j].group, view_colours[i].group);
             j++)
            (*count)++;

        return i;
    }

    *count = 0;

    return -1;
}

/* The pixel formats the export's format row offers, and how many.
 *
 * A linear scene-referred export has the family's float formats, half and
 * single, which are the only things the direct path writes linear light
 * in; ACEScct has the family's own list, under ten bits greyed. Anything
 * else is what the probe found the family's encoder opens with; a format
 * it does not open with is never offered at all, and one it opens with
 * that the preset's profile forbids is offered greyed. */
static int view_menu_fmts(const struct view_state *st,
                          const enum AVPixelFormat **out)
{
    const struct view_preset_sel *s = &st->look.video_preset;

    if (s->family < 0 || s->family >= view_nb_preset_families) {
        *out = NULL;
        return 0;
    }

    if (view_colours[CRAW_CLAMP(s->colour, 0, VIEW_NB_COLOURS - 1)].needs_deep_rgb)
        return view_presets_scene_fmts(s->family, out);

    return view_presets_fmts(s->family, out);
}

/* A leaf column's entries select rather than opening another column. */
static bool view_menu_leaf(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
    int first, n;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        return true;
    case VIEW_MENU_PRESET:
        return col == 1;
    case VIEW_MENU_COLOUR:
        /* A group with one entry in it is its own leaf: BT.709 and the two
         * ACES spaces are one row each, BT.2020 opens its transfers. */
        if (col == 1)
            return true;
        first = m->sel[0] >= 0 ? view_menu_colour_group(m->sel[0], &n) : -1;
        return first < 0 || n <= 1;
    default:
        return (m->sel[0] == 1 && col == 2) || (m->sel[0] == 0 && col == 3);
    }
}

static int view_menu_count(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
#endif
    int n;

    if (m->kind == VIEW_MENU_PIXFMT) {
        const enum AVPixelFormat *fmts;

        return col == 0 ? view_menu_fmts(st, &fmts) : 0;
    }

    if (m->kind == VIEW_MENU_PRESET) {
        if (col == 0)
            return view_nb_preset_families;
        if (col == 1 && m->sel[0] >= 0 &&
            m->sel[0] < view_nb_preset_families)
            return view_preset_families[m->sel[0]].nb_presets;
        return 0;
    }

    if (m->kind == VIEW_MENU_COLOUR) {
        if (col == 0)
            return view_menu_colour_groups();
        if (col == 1 && m->sel[0] >= 0) {
            view_menu_colour_group(m->sel[0], &n);
            return n > 1 ? n : 0;
        }
        return 0;
    }

#if CINEVIEW_HAVE_LUTS
    switch (col) {
    case 0:
        return 2;
    case 1:
        if (m->sel[0] < 0)
            return 0;
        return m->sel[0] == 0 ? view_luts_film_category_count()
                              : view_luts_look_category_count();
    case 2:
        if (m->sel[0] < 0 || m->sel[1] < 0)
            return 0;
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(m->sel[1]);
            return fc ? fc->n_films : 0;
        }
        lc = view_lut_look_category(m->sel[1]);
        return lc ? lc->n_luts : 0;
    case 3:
        f = view_menu_film(st);
        return f ? f->n_luts : 0;
    default:
        return 0;
    }
#else
    (void)col;
    (void)n;
    return 0;
#endif
}

/* What an entry of a leaf column stands for, or -1: a rendering for the
 * film menu, a preset for the codec one, a colour space for the third. */
static int view_menu_lut(const struct view_state *st, int col, int i)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
#endif
    int first, n;

    if (!view_menu_leaf(st, col) || i < 0 || i >= view_menu_count(st, col))
        return -1;

    if (m->kind == VIEW_MENU_PRESET || m->kind == VIEW_MENU_PIXFMT)
        return i;

    if (m->kind == VIEW_MENU_COLOUR) {
        first = view_menu_colour_group(col == 0 ? i : m->sel[0], &n);
        if (first < 0)
            return -1;
        return col == 0 ? first : first + i;
    }

#if CINEVIEW_HAVE_LUTS
    if (col == 2) {
        lc = view_lut_look_category(m->sel[1]);
        return lc && lc->luts ? (int)lc->luts[i] : -1;
    }

    f = view_menu_film(st);
    return f ? f->first_lut + i : -1;
#else
    return -1;
#endif
}

/* Why the entry cannot be chosen, or NULL. A greyed row still shows and
 * still highlights: what it says is the point. */
static const char *view_menu_why_not(const struct view_state *st, int col, int i)
{
    const struct view_lut_menu *m = &st->menu;
    const struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    int ix, cnt;

    if (m->kind == VIEW_MENU_PIXFMT) {
        const char *why;

        if (i < 0 || i >= view_menu_fmts(st, &fmts))
            return NULL;
        why = view_presets_fmt_why_not(s->family, s->preset, fmts[i]);
        return why ? why : view_presets_colour_fmt_why_not(s->colour, fmts[i]);
    }

    if (m->kind == VIEW_MENU_PRESET)
        return col == 0 ? view_presets_family_why_not(i)
                        : view_presets_preset_why_not(m->sel[0], i);

    if (m->kind == VIEW_MENU_COLOUR) {
        ix = view_menu_lut(st, col, i);
        if (ix < 0 && col == 0)
            ix = view_menu_colour_group(i, &cnt);
        return ix < 0 ? NULL : view_presets_sel_colour_why_not(s, ix);
    }

    return NULL;
}

static void view_menu_label(const struct view_state *st, int col, int i,
                            char *buf, size_t n)
{
    const struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLutFilmCategory *fc;
    const ViewLutLookCategory *lc;
    const ViewLutFilm *f;
    const ViewLut *l;
#endif
    const enum AVPixelFormat *fmts;
    int ix;

    buf[0] = '\0';

    if (m->kind == VIEW_MENU_PIXFMT) {
        if (i >= 0 && i < view_menu_fmts(st, &fmts))
            view_presets_fmt_label(fmts[i], buf, n);
        return;
    }

    if (m->kind == VIEW_MENU_PRESET) {
        if (col == 0 && i < view_nb_preset_families)
            snprintf(buf, n, "%s", view_preset_families[i].name);
        else if (col == 1 && m->sel[0] >= 0 &&
                 m->sel[0] < view_nb_preset_families &&
                 i < view_preset_families[m->sel[0]].nb_presets)
            snprintf(buf, n, "%s",
                     view_preset_families[m->sel[0]].presets[i].name);
        return;
    }

    if (m->kind == VIEW_MENU_COLOUR) {
        if (col == 0) {
            int cnt;

            ix = view_menu_colour_group(i, &cnt);
            if (ix >= 0)
                snprintf(buf, n, "%s", view_colours[ix].group);
        } else {
            ix = view_menu_lut(st, col, i);
            if (ix >= 0)
                snprintf(buf, n, "%s", view_colours[ix].name);
        }
        return;
    }

#if CINEVIEW_HAVE_LUTS
    switch (col) {
    case 0:
        snprintf(buf, n, "%s", i == 0 ? "Films" : "Looks");
        return;

    case 1:
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(i);
            if (fc)
                snprintf(buf, n, "%s", fc->name);
        } else {
            lc = view_lut_look_category(i);
            if (lc)
                snprintf(buf, n, "%s", lc->name);
        }
        return;

    case 2:
        if (m->sel[0] == 0) {
            fc = view_lut_film_category(m->sel[1]);
            f = fc ? view_lut_film(fc->first_film + i) : NULL;
            if (f)
                snprintf(buf, n, "%s", f->name);
            return;
        }
        /* A look cuts across films, so without the stock on the row the
         * rendering's name says nothing about what it is emulating. */
        l = view_lut_get(view_menu_lut(st, col, i));
        f = l ? view_lut_film(l->film) : NULL;
        if (l && f)
            snprintf(buf, n, "%s \xc2\xb7 %s", l->name, f->name);
        return;

    case 3:
        l = view_lut_get(view_menu_lut(st, col, i));
        if (l && l->label[0])
            snprintf(buf, n, "%s \xc2\xb7 %s", l->name, l->label);
        else if (l)
            snprintf(buf, n, "%s", l->name);
        return;

    default:
        return;
    }
#else
    (void)col;
#endif
}

/* What choosing entry `i` of leaf column `col` does. */
static void view_menu_select(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;
    struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    int ix = view_menu_lut(st, col, i);

    if (ix < 0 || view_menu_why_not(st, col, i))
        return;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        if (ix >= view_menu_fmts(st, &fmts))
            break;
        if (view_colours[CRAW_CLAMP(s->colour, 0, VIEW_NB_COLOURS - 1)].needs_deep_rgb) {
            s->scene_pix_fmt = fmts[ix];
            s->set = true;
            break;
        }
        s->pix_fmt = fmts[ix];
        /* Stamped with the family it was chosen under: the row falls back
         * to the preset's own pin the moment the family moves, since the
         * next family's encoder need not take this format at all. */
        s->pix_fmt_family = s->family;
        s->set = true;
        break;
    case VIEW_MENU_PRESET:
        view_presets_take_preset(s, m->sel[0], ix);
        break;
    case VIEW_MENU_COLOUR:
        view_presets_take_colour(s, ix);
        break;
    default:
#if CINEVIEW_HAVE_LUTS
        view_lut_select_film(st, ix);
#endif
        break;
    }
}

/* The entry that is in use, for the mark the drawing puts behind it. */
static int view_menu_current(const struct view_state *st, int col)
{
    const struct view_lut_menu *m = &st->menu;
    const struct view_preset_sel *s = &st->look.video_preset;
    const enum AVPixelFormat *fmts;
    enum AVPixelFormat cur;
    int first, n;

    switch (m->kind) {
    case VIEW_MENU_PIXFMT:
        n = view_menu_fmts(st, &fmts);
        cur = view_presets_sel_out_fmt(s);
        for (int i = 0; i < n; i++)
            if (fmts[i] == cur)
                return i;
        return -1;
    case VIEW_MENU_PRESET:
        return col == 0 ? s->family : (m->sel[0] == s->family ? s->preset : -1);
    case VIEW_MENU_COLOUR:
        if (col == 0) {
            for (int g = 0; g < view_menu_colour_groups(); g++) {
                first = view_menu_colour_group(g, &n);
                if (first >= 0 && s->colour >= first && s->colour < first + n)
                    return g;
            }
            return -1;
        }
        first = m->sel[0] >= 0 ? view_menu_colour_group(m->sel[0], &n) : -1;
        return first >= 0 && s->colour >= first && s->colour < first + n
             ? s->colour - first : -1;
    default:
        /* A rendering is only findable by walking the leaf column, since
         * the index says nothing about which entry holds it. */
        if (st->look.film_sim < 0 || !view_menu_leaf(st, col))
            return -1;
        n = view_menu_count(st, col);
        for (int i = 0; i < n; i++)
            if (view_menu_lut(st, col, i) == st->look.film_sim)
                return i;
        return -1;
    }
}

static float view_menu_width(const struct view_state *st, int col)
{
    int n = view_menu_count(st, col);
    float text = 0.0f, w;

    for (int i = 0; i < n; i++) {
        char buf[192];

        view_menu_label(st, col, i, buf, sizeof(buf));
        text = CRAW_MAX(text, view_text_w(buf));
    }

    w = text + 2 * VIEW_MENU_XPAD + 6.0f +
        (view_menu_leaf(st, col) ? 0.0f : VIEW_MENU_ARROW);

    return CRAW_CLAMP(w, VIEW_MENU_MIN_W, VIEW_MENU_MAX_W);
}

/* Places the columns. The first hangs off the panel row, each next one
 * beside the entry that opened it, and none of them may leave the window:
 * the cascade runs left when there is no room on the right, and a column
 * taller than the window scrolls instead of growing. */
static void view_menu_layout(struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    float winw = (float)st->gpu.ui_w;
    float winh = (float)st->gpu.ui_h;
    const float margin = VIEW_UI_MARGIN;
    int dir;

    /* How far the highlighted path reaches. A column opens only once its
     * parent entry is highlighted and actually has children. */
    m->cols = 1;
    for (int c = 0; c + 1 < VIEW_MENU_COLS; c++) {
        if (m->sel[c] < 0 || view_menu_leaf(st, c) ||
            view_menu_count(st, c + 1) <= 0)
            break;
        m->cols = c + 2;
    }

    dir = (m->ax + m->aw + view_menu_width(st, 0) <= winw - margin) ? 1 : -1;
    /* The pointer pass needs it too: "towards the submenu" is whichever
     * way the cascade runs. */
    m->dir = dir;

    for (int c = 0; c < m->cols; c++) {
        float w = view_menu_width(st, c);
        float full = (float)view_menu_count(st, c) * VIEW_MENU_ROW +
                     2 * VIEW_MENU_PAD;
        float h = CRAW_MIN(full, winh - 2 * margin);
        float maxs = CRAW_MAX(0.0f, full - h);
        float s, x, y;

        if (c == 0) {
            x = dir > 0 ? m->ax + m->aw : m->ax - w;
            y = m->ay;
        } else {
            x = dir > 0 ? m->x[c - 1] + m->w[c - 1] : m->x[c - 1] - w;
            /* Line the column up with the row that opened it, the way a
             * submenu hangs off its parent entry. */
            y = m->y[c - 1] + VIEW_MENU_PAD - m->scroll[c - 1] +
                (float)m->sel[c - 1] * VIEW_MENU_ROW;
        }

        m->x[c] = CRAW_CLAMP(x, margin, CRAW_MAX(margin, winw - margin - w));
        m->y[c] = CRAW_CLAMP(y, margin, CRAW_MAX(margin, winh - margin - h));
        m->w[c] = w;
        m->h[c] = h;

        s = CRAW_CLAMP(m->scroll[c], 0.0f, maxs);
        /* A column opened on the current selection has to show it, or the
         * menu opens scrolled to the top of a list twenty-six entries long
         * with the highlighted one somewhere below the fold. */
        if (m->sel[c] >= 0) {
            float top = (float)m->sel[c] * VIEW_MENU_ROW;
            float inner = h - 2 * VIEW_MENU_PAD;

            if (top < s)
                s = top;
            else if (top + VIEW_MENU_ROW > s + inner)
                s = top + VIEW_MENU_ROW - inner;
            s = CRAW_CLAMP(s, 0.0f, maxs);
        }
        m->scroll[c] = s;
    }
}

static ImRect view_menu_column(const struct view_lut_menu *m, int c)
{
    return ImRect(m->x[c], m->y[c], m->x[c] + m->w[c], m->y[c] + m->h[c]);
}

/* The rectangle of entry `i` of column `c`, in window pixels. */
static ImRect view_menu_entry(const struct view_lut_menu *m, int c, int i)
{
    float x = m->x[c] + 1.0f;
    float y = m->y[c] + VIEW_MENU_PAD + (float)i * VIEW_MENU_ROW - m->scroll[c];

    return ImRect(x, y, x + m->w[c] - 2.0f, y + VIEW_MENU_ROW);
}

static void view_menu_descend(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;

    if (m->sel[col] == i)
        return;

    m->sel[col] = i;
    /* A different parent invalidates everything below it. */
    for (int k = col + 1; k < VIEW_MENU_COLS; k++) {
        m->sel[k] = -1;
        m->scroll[k] = 0.0f;
    }
}

/* A click on a row that opens another column: the film menu just walks
 * into it, but the export's two treat the parent as a choice of its own -
 * clicking a family takes that family's default preset, clicking a colour
 * group takes the first entry under it - which is what the row above the
 * menu says it does. Returns true when that settled it. */
static bool view_menu_click_parent(struct view_state *st, int col, int i)
{
    struct view_lut_menu *m = &st->menu;
    struct view_preset_sel *s = &st->look.video_preset;
    int first, n;

    if (m->kind == VIEW_MENU_PRESET && col == 0) {
        if (view_presets_family_why_not(i))
            return true;            /* greyed: eat the click, change nothing */
        view_presets_take_preset(s, i, view_preset_families[i].def);
        return true;
    }

    if (m->kind == VIEW_MENU_COLOUR && col == 0) {
        first = view_menu_colour_group(i, &n);
        if (first >= 0)
            view_presets_take_colour(s, first);
        return true;
    }

    return false;
}

/* Pointer pass: hover moves the highlight and opens the next column, a
 * click selects a rendering or opens a submenu at once, the wheel scrolls
 * the column under the pointer. Returns 1 when a choice was made.
 *
 * Hover is deliberately not immediate. Sweeping the pointer diagonally
 * from an entry to something in the submenu it opened crosses that entry's
 * siblings, and a menu that switched on every crossing would collapse the
 * column being aimed at - which is what made this feel finicky. So an
 * entry has to be rested on before its submenu opens, and for a moment
 * after one opens the siblings are ignored while the pointer is still
 * travelling towards it. */
static int view_menu_pointer(struct view_state *st, bool *over)
{
    struct view_lut_menu *m = &st->menu;
    const ImGuiIO &io = ImGui::GetIO();
    ImVec2 p = io.MousePos;
    int64_t now = av_gettime_relative();
    float dx = p.x - m->prev_x;
    int hit_col = -1, hit_entry = -1;
    bool clicked = false;
    int pick = -1;

    *over = false;

    for (int c = 0; c < m->cols; c++) {
        ImRect r = view_menu_column(m, c);
        int n = view_menu_count(st, c);

        if (!r.Contains(p))
            continue;
        *over = true;

        /* A column taller than the window takes the wheel itself, so the
         * picture underneath neither zooms nor scrolls. */
        if (io.MouseWheel != 0.0f) {
            float full = (float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD;

            m->scroll[c] = CRAW_CLAMP(m->scroll[c] - io.MouseWheel * VIEW_MENU_ROW * 2.0f,
                                      0.0f, CRAW_MAX(0.0f, full - m->h[c]));
        }

        for (int i = 0; i < n; i++) {
            ImRect e = view_menu_entry(m, c, i);

            if (p.y < e.Min.y || p.y >= e.Max.y)
                continue;
            /* Only rows fully inside the column count, or the half row
             * peeking out of a scrolled list would take clicks. */
            if (e.Min.y < r.Min.y || e.Max.y > r.Max.y)
                break;

            hit_col = c;
            hit_entry = i;
            clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                      e.Contains(io.MouseClickedPos[ImGuiMouseButton_Left]);
            break;
        }
    }

    /* How long the pointer has been where it is. */
    if (hit_col != m->hover_col || hit_entry != m->hover_entry) {
        m->hover_col = hit_col;
        m->hover_entry = hit_entry;
        m->hover_since = now;
    }
    m->guarded = false;

    if (hit_col >= 0) {
        bool leaf = view_menu_leaf(st, hit_col);

        if (clicked) {
            /* A click is unambiguous: act on it at once, whatever the
             * hover timers were about to do. */
            if (leaf) {
                view_menu_select(st, hit_col, hit_entry);
                pick = 1;
            } else if (view_menu_click_parent(st, hit_col, hit_entry)) {
                pick = 1;
            } else {
                view_menu_descend(st, hit_col, hit_entry);
                m->guard_col = hit_col;
                m->guard_until = now + VIEW_MENU_GUARD_US;
            }
        } else if (m->sel[hit_col] != hit_entry) {
            /* Still travelling towards the submenu this column opened:
             * the siblings on the way do not get to close it. */
            bool guarded = hit_col == m->guard_col && now < m->guard_until &&
                           dx * (float)m->dir > 0.0f;

            m->guarded = guarded;
            if (leaf) {
                /* Nothing to protect and nothing to open, so a leaf's
                 * highlight simply follows the pointer. */
                view_menu_descend(st, hit_col, hit_entry);
            } else if (!guarded && now - m->hover_since >= VIEW_MENU_OPEN_US) {
                view_menu_descend(st, hit_col, hit_entry);
                m->guard_col = hit_col;
                m->guard_until = now + VIEW_MENU_GUARD_US;
            }
        }
    }

    m->prev_x = p.x;
    m->prev_y = p.y;

    return pick;
}

bool view_ui_menu_due(const struct view_state *st, int64_t now)
{
    const struct view_lut_menu *m = &st->menu;

    if (!m->open || m->hover_col < 0 || m->hover_entry < 0 ||
        view_menu_leaf(st, m->hover_col))
        return false;

#if CINEVIEW_HAVE_LUTS
    /* A film's preview, due a moment after the pointer stopped on it and
     * not yet shown. */
    if (m->kind == VIEW_MENU_FILM && m->sel[0] == 0 && m->hover_col == 2 &&
        !m->guarded && now - m->hover_since >= VIEW_MENU_PEEK_US) {
        int ix = view_menu_film_base(view_menu_film_at(st, m->hover_entry));

        if (ix >= 0 && ix != st->preview_sim && ix != st->look.film_sim)
            return true;
    }
#endif

    if (m->sel[m->hover_col] == m->hover_entry)
        return false;
    if (m->guarded)
        return now >= m->guard_until;
    return now - m->hover_since >= VIEW_MENU_OPEN_US;
}

static void view_menu_draw(struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    ImFont *font = ImGui::GetFont();
    float fs = ImGui::GetFontSize();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, view_col_surface);

    for (int c = 0; c < m->cols; c++) {
        ImRect r = view_menu_column(m, c);
        int n = view_menu_count(st, c);
        int cur = view_menu_current(st, c);
        bool leaf = view_menu_leaf(st, c);
        char name[32];

        snprintf(name, sizeof(name), "##cineview-lutmenu-%d", c);
        ImGui::SetNextWindowPos(r.Min);
        ImGui::SetNextWindowSize(r.GetSize());
        if (!ImGui::Begin(name, NULL,
                          ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                          ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                          ImGuiWindowFlags_NoFocusOnAppearing |
                          ImGuiWindowFlags_NoScrollWithMouse)) {
            ImGui::End();
            continue;
        }

        ImDrawList *out = ImGui::GetWindowDrawList();

        for (int i = 0; i < n; i++) {
            ImRect e = view_menu_entry(m, c, i);
            char buf[192];
            bool on = m->sel[c] == i;
            bool grey = view_menu_why_not(st, c, i) != NULL;
            float ty = e.Min.y + (VIEW_MENU_ROW - fs) * 0.5f;

            if (e.Max.y <= r.Min.y || e.Min.y >= r.Max.y)
                continue;           /* scrolled out of the column */

            view_menu_label(st, c, i, buf, sizeof(buf));
            /* The highlight, and the entry actually in use, which gets a
             * bar of the accent at its left edge so that it stays findable
             * while the highlight is off wandering. */
            if (on)
                out->AddRectFilled(e.Min, e.Max, ImGui::GetColorU32(view_col_accent_dim));
            if (i == cur)
                out->AddRectFilled(e.Min, ImVec2(e.Min.x + 3.0f, e.Max.y),
                                   ImGui::GetColorU32(view_col_accent));

            ImVec4 clip(e.Min.x + VIEW_MENU_XPAD, e.Min.y,
                        e.Max.x - VIEW_MENU_XPAD - (leaf ? 0.0f : VIEW_MENU_ARROW),
                        e.Max.y);
            out->AddText(i == cur ? view_font_strong : font, fs, ImVec2(clip.x, ty),
                         ImGui::GetColorU32(grey ? view_col_text_faint : view_col_text),
                         buf, NULL, 0.0f, &clip);

            /* The mark that says this entry opens another column, pointing
             * the way the cascade runs. */
            if (!leaf)
                ImGui::RenderArrow(out, ImVec2(e.Max.x - VIEW_MENU_ARROW, ty),
                                   ImGui::GetColorU32(view_col_text_dim),
                                   m->dir < 0 ? ImGuiDir_Left : ImGuiDir_Right, 0.6f);
        }

        /* A scrolled column says so, or a list cut off at the window edge
         * looks like the whole list. */
        if ((float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD > r.GetHeight()) {
            float full = (float)n * VIEW_MENU_ROW + 2 * VIEW_MENU_PAD;
            float y0 = r.Min.y + r.GetHeight() * (m->scroll[c] / full);

            out->AddRectFilled(ImVec2(r.Max.x - 4.0f, y0),
                               ImVec2(r.Max.x - 1.0f,
                                      y0 + r.GetHeight() * (r.GetHeight() / full)),
                               ImGui::GetColorU32(view_col_frame_act));
        }

        ImGui::End();
    }

    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
}

void view_ui_menu_open(struct view_state *st, enum view_menu_kind kind)
{
    struct view_lut_menu *m = &st->menu;
#if CINEVIEW_HAVE_LUTS
    const ViewLut *l = view_lut_get(st->look.film_sim);
#endif

    if (m->open)
        return;

    /* The film simulation is one of the three things a scene-referred
     * export takes away, and a cascade that opened over a greyed row would
     * be offering a choice that changes neither the file nor the screen.
     * A locked panel has no rows a cascade could change anything from. */
    if (kind == VIEW_MENU_FILM && view_export_is_scene(st))
        return;
    if (view_ui_locked(st))
        return;

    m->open = true;
    m->kind = kind;
    m->cols = 1;
    for (int c = 0; c < VIEW_MENU_COLS; c++) {
        m->sel[c] = -1;
        m->scroll[c] = 0.0f;
    }

    /* No stale hover timers from the last time it was open, and no guard
     * protecting a column that no longer exists. */
    m->hover_col = -1;
    m->hover_entry = -1;
    m->hover_since = av_gettime_relative();
    m->guard_col = -1;
    m->guard_until = 0;

    /* Open on what is already selected, so the columns come up showing
     * the current setting rather than the top of the list. */
    if (kind == VIEW_MENU_PRESET) {
        m->sel[0] = st->look.video_preset.family;
        m->sel[1] = st->look.video_preset.preset;
        return;
    }
    if (kind == VIEW_MENU_COLOUR) {
        m->sel[0] = view_menu_current(st, 0);
        m->sel[1] = m->sel[0] >= 0 ? view_menu_current(st, 1) : -1;
        return;
    }
    if (kind == VIEW_MENU_PIXFMT) {
        m->sel[0] = view_menu_current(st, 0);
        return;
    }

#if CINEVIEW_HAVE_LUTS
    if (l) {
        const ViewLutFilm *f = view_lut_film(l->film);
        const ViewLutFilmCategory *c = f ? view_lut_film_category(f->category) : NULL;

        if (f && c) {
            m->sel[0] = 0;
            m->sel[1] = f->category;
            m->sel[2] = 0;
            for (int i = 0; i < c->n_films; i++) {
                const ViewLutFilm *g = view_lut_film(c->first_film + i);

                if (g == f) {
                    m->sel[2] = i;
                    break;
                }
            }
            m->sel[3] = st->look.film_sim - f->first_lut;
        }
    }
#endif
}

void view_ui_menu_close(struct view_state *st)
{
    st->menu.open = false;
    st->menu.cols = 0;
    /* A pick has already taken the preview over; anything else closing
     * the menu puts the chosen rendering back. */
    view_lut_preview(st, -1);
}

bool view_ui_menu_covers(const struct view_state *st, float x, float y)
{
    const struct view_lut_menu *m = &st->menu;
    float sx = st->win_w > 0 ? (float)st->gpu.ui_w / (float)st->win_w : 1.0f;
    float sy = st->win_h > 0 ? (float)st->gpu.ui_h / (float)st->win_h : 1.0f;

    if (!m->open)
        return false;

    /* SDL reports window points, the panel and the menu live in drawable
     * pixels; the two differ under a fractional scale. */
    for (int c = 0; c < m->cols; c++)
        if (view_menu_column(m, c).Contains(ImVec2(x * sx, y * sy)))
            return true;

    return false;
}

/* One panel row that opens a cascade: its label, and a field showing what
 * is selected with a chevron pointing the way the columns open. The menu
 * itself is built after the panel so that it draws on top. */
static void view_menu_row(struct view_state *st, enum view_menu_kind kind,
                          const char *label, const char *value)
{
    struct view_lut_menu *m = &st->menu;
    const ImGuiStyle &style = ImGui::GetStyle();
    bool open = m->open && m->kind == kind;
    float fs = ImGui::GetFontSize();
    char shown[192], id[224];
    float cw = view_prop(label);
    bool pressed;

    view_fit_text(cw - 2.0f * style.FramePadding.x - fs, value, shown, sizeof(shown));
    /* The id stays the row's whatever the field says. */
    snprintf(id, sizeof(id), "%s###menu%d", shown, (int)kind);

    /* Click to toggle, never hover: a menu that dropped open whenever the
     * pointer crossed the row on its way somewhere else was the single
     * most irritating thing about it. */
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
    ImGui::PushStyleColor(ImGuiCol_Button, open ? view_col_frame_act : view_col_frame);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, view_col_frame_hi);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, view_col_frame_act);
    pressed = ImGui::Button(id, ImVec2(cw, 0.0f));
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();

    ImGui::RenderArrow(ImGui::GetWindowDrawList(),
                       ImVec2(ImGui::GetItemRectMax().x - style.FramePadding.x - fs * 0.6f,
                              ImGui::GetItemRectMin().y + style.FramePadding.y + fs * 0.2f),
                       ImGui::GetColorU32(open ? view_col_accent : view_col_text_dim),
                       ImGuiDir_Left, 0.6f);

    if (pressed) {
        if (open) {
            view_ui_menu_close(st);
        } else {
            view_ui_menu_close(st);
            view_ui_menu_open(st, kind);
        }
    }

    /* The anchor is the panel's whole width rather than the field's, so
     * the cascade opens flush with the panel's edge instead of halfway
     * across it. It is only written by whichever row owns the open menu,
     * or two rows would fight over where the columns go. */
    if (m->open && m->kind == kind) {
        m->ax = ImGui::GetWindowPos().x;
        m->aw = ImGui::GetWindowSize().x;
        m->ay = ImGui::GetItemRectMin().y;
        m->ah = ImGui::GetItemRectSize().y;
        m->anchored = true;
    }
}

static void view_panel_film_sim(struct view_state *st)
{
    view_menu_row(st, VIEW_MENU_FILM, "Film simulation",
#if CINEVIEW_HAVE_LUTS
                  view_lut_film_name(st));
#else
                  "not built in");
#endif
}

/* Runs the whole menu for this frame: pointer first so that the cascade
 * reacts in the same frame the pointer moved, then the layout it implies,
 * then the drawing. */
static void view_menu_frame(struct view_state *st)
{
    struct view_lut_menu *m = &st->menu;
    const ImGuiIO &io = ImGui::GetIO();
    bool over = false;
    int pick;

    if (!m->open)
        return;

    /* The row the menu hangs off was not emitted this frame - its section
     * has been collapsed under it - so there is nothing to hang off and
     * nothing to draw. Closing rather than hiding, or Escape would go on
     * being swallowed by a menu nobody can see. A locked panel closes
     * it for the same reason: the columns do their own hit testing, and a
     * choice made in one would land on a setting the run has copied. */
    if (!m->anchored || view_ui_locked(st)) {
        view_ui_menu_close(st);
        return;
    }

    pick = view_menu_pointer(st, &over);
    if (pick > 0) {
        view_ui_menu_close(st);
        return;
    }

    /* Merely moving the pointer off the columns does not close anything:
     * the menu is opened by a click and it stays until something says
     * otherwise. A click outside it is one of those things, and so is the
     * row itself, which toggles and is therefore left alone here. */
    if (!over && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
        !ImRect(m->ax, m->ay, m->ax + m->aw, m->ay + m->ah)
             .Contains(io.MouseClickedPos[ImGuiMouseButton_Left])) {
        view_ui_menu_close(st);
        return;
    }

    view_menu_layout(st);

    /* The film menu's highlight is a preview: whatever rendering the leaf
     * column's rests on is shown while it does. The leaf is the last
     * column open, and view_menu_lut() answers -1 for anything that is
     * not a rendering - a parent column, or a leaf with nothing
     * highlighted yet. A film with none of its renderings highlighted
     * previews as its base, so the films column can be walked without
     * going into every submenu; a look category has no such rendering,
     * and previews as nothing. */
    if (m->kind == VIEW_MENU_FILM) {
        int ix = view_menu_lut(st, m->cols - 1, m->sel[m->cols - 1]);

#if CINEVIEW_HAVE_LUTS
        /* A film under the pointer previews as its base once the pointer
         * has been on it a moment (VIEW_MENU_PEEK_US), ahead of the rest
         * that opens its column: the preview is what the hover is for, and
         * a rendering unpacks in a few milliseconds. Waiting for the column
         * meant the rest, then the idle tick that noticed it, then a
         * present - a third of a second between the pointer stopping on a
         * film and its look, where a variant showed in a tenth. The moment
         * is what keeps the films a hand crosses on its way into an open
         * column from flashing by, as is the guard. */
        if (m->sel[0] == 0 && m->hover_col == 2 && m->hover_entry >= 0 &&
            !m->guarded &&
            av_gettime_relative() - m->hover_since >= VIEW_MENU_PEEK_US)
            ix = view_menu_film_base(view_menu_film_at(st, m->hover_entry));
        else if (ix < 0)
            ix = view_menu_film_base(view_menu_film(st));
#endif
        view_lut_preview(st, ix);
    }

    view_menu_draw(st);
}

/* The rate row under the scrub bar, DESIGN.md 23: the clip shown and
 * written at a lower fixed rate, frame for frame, so that a 59.94 clip at
 * 23.976 plays two and a half times slower and exports as a 23.976 file two
 * and a half times as long. A slider over the stops - the fixed rates below
 * the clip's own, slowest at the left, and the clip's own at the right -
 * with what the stop means written on it. It goes where it is pressed, so
 * the knob at either end is as easy to pick up as anywhere else.
 *
 * Moving it chooses the rate and nothing else: no seek, no pause. The run
 * loop makes the choice the rate, and does what that means for the sound
 * and the clock, in one step. A clip at 23.976 has nothing slower to go
 * to, and its row is greyed and full, where the clip's own rate is. With
 * every other frame the stops stay the clip's, and the half rate it then
 * plays at its own speed sits among them, so the knob can go either side
 * of it (DESIGN.md 23.6). The every-other-frame box ends the row. */
static void view_panel_rate(struct view_state *st)
{
    const ImGuiStyle &style = ImGui::GetStyle();
    AVRational stops[VIEW_RATE_MAX_STOPS];
    char text[64], fmt[132];
    int own = 0, n = view_rate_stops(st, stops, VIEW_RATE_MAX_STOPS, &own);
    int cur = own, pos, half = st->half;
    float cw;

    if (view_slowed(st))
        for (int i = 0; i < n; i++)
            if (i != own && !av_cmp_q(stops[i], st->rate))
                cur = i;

    cw = view_prop("Rate");

    /* The choice, and only the choice: the run loop makes it the rate
     * before the clock is next read (view_rate_follow()). */
    pos = cur;
    view_rate_label(st, stops[CRAW_CLAMP(pos, 0, CRAW_MAX(n - 1, 0))], text, sizeof(text));
    view_fmt_escape(text, fmt, sizeof(fmt));
    ImGui::SetNextItemWidth(cw - ImGui::GetFrameHeight() - style.ItemSpacing.x);
    if (n < 2) {
        int one = 1, lo = 0, hi = 1;

        ImGui::BeginDisabled();
        view_slider_scalar("##rate", ImGuiDataType_S32, &one, &lo, &hi, fmt,
                           ImGuiSliderFlags_NoInput);
        ImGui::EndDisabled();
    } else {
        int lo = 0, hi = n - 1;

        if (view_slider_scalar("##rate", ImGuiDataType_S32, &pos, &lo, &hi, fmt,
                               ImGuiSliderFlags_NoInput) && pos != cur)
            st->rate_choice = pos == own ? AVRational { 0, 0 } : stops[pos];
    }

    /* Every frame, or every other one: the odd-numbered or the
     * even-numbered, counted from frame 0 as the readout counts. A
     * checkbox with three states, moved on by each click as by `f`: empty
     * for all frames, its upper left half filled for the odd ones and its
     * lower right half for the even ones. The slider says which ("14.99
     * fps, odd frames"). The choice and nothing else: the run loop takes
     * it up with the rate (view_rate_follow()). Live whatever the slider
     * is, since leaving frames out is what gives a clip with nothing
     * slower than its own rate stops to move to. */
    ImGui::SameLine();
    if (view_check3("##half", &half))
        st->half = (enum view_half)half;
}

/* The scrub bar: the clip as a bar filled up to the frame on screen. A
 * press anywhere on it is the frame under the pointer, the picture stays
 * on it for as long as the button is held, and the pointer moving while
 * held takes it along - also off the bar, a drag that wanders keeps
 * scrubbing. Playback stops while the bar is held and goes on from
 * wherever it is let go if the clip was playing before. */
static void view_scrub_bar(struct view_state *st, int64_t total)
{
    const int64_t first = 0, last = total - 1;
    int64_t want = CRAW_CLAMP(st->index, 0, total - 1);
    bool held;

    ImGui::SetNextItemWidth(-FLT_MIN);
    view_slider_scalar("##scrub", ImGuiDataType_S64, &want, &first, &last, "",
                       ImGuiSliderFlags_NoInput);
    held = ImGui::IsItemActive();

    if (held) {
        if (!st->scrubbing) {
            st->scrubbing = true;
            st->scrub_resume = st->playing;
            st->scrub_want = st->seek_to >= 0 ? st->seek_to : st->index;
            st->playing = false;
            view_decoder_readahead(&st->dec, 2);
        }
        if (want != st->scrub_want) {
            st->scrub_want = want;
            view_log(st, "scrub: frame %" PRId64 "\n", want);
            st->seek_to = want;
            st->need_frame = true;
        }
    }

    /* Let go. A drag was a stream of seeks, and each would have paid for
     * the temporal denoise's pre-roll (view_step_frame()), which halves the
     * rate the picture follows the pointer at; they started the average
     * cold instead, and the frame let go on is sought once more, with its
     * pre-roll. */
    if (st->scrubbing && !held) {
        st->scrubbing = false;
        view_decoder_readahead(&st->dec, VIEW_PKTQ_MAX);
        if (st->scrub_resume)
            st->playing = true;
        st->scrub_resume = false;
        if (st->seek_to < 0 && view_temporal_strength(st) > 0.0f) {
            st->seek_to = st->index;
            st->seek_cold = true;
            st->need_frame = true;
        }
    }
}

static void view_panel_transport(struct view_state *st)
{
    int64_t total = st->dec.nb_frames;

    /* No transport buttons: the keyboard is the transport. Space plays and
     * pauses, Shift with Page Up and Page Down are the ends of the clip,
     * Left and Right step five seconds and Shift with them one. The scrub
     * bar stays, because going to a frame by eye is the one thing keys are
     * bad at, and under it the rate, which no key has. */
    if (total > 1)
        view_scrub_bar(st, total);

    /* A still is not played, so there is nothing for it to be slowed to,
     * nor frames for it to leave out. */
    if (st->dec.fmt && !st->dec.still)
        view_panel_rate(st);
}

/* One section for both kinds of clip. A raw clip is balanced by libcraw
 * from the sensor's ratios and the camera's tables; a developed one is
 * rebalanced from the temperature it records, on a locus of its own in its
 * own linear RGB (videowb.c), which gives it the same three modes, the same
 * pick and the same two numbers in the same units. Only the developed one
 * has a switch - a raw clip's balance is always applied - and it lights
 * when anything here is touched. */
static void view_panel_wb(struct view_state *st)
{
    static const char *const modes[] = { "As shot", "Auto", "Manual" };
    static const enum view_wb mode_of[] = { VIEW_WB_AS_SHOT, VIEW_WB_AUTO, VIEW_WB_MANUAL };
    struct view_look *lk = &st->look;
    const enum view_wb mode = lk->wb;
    const CRAWIlluminant manual = lk->manual;
    const float ambience = st->awb.params.ambience;
    int sel = 0;

    for (int i = 0; i < 3; i++)
        if (mode_of[i] == lk->wb)
            sel = i;

    view_prop("Mode");
    if (view_segmented("wbmode", modes, 3, &sel)) {
        /* Manual starts from the balance on screen, whichever mode put it
         * there, so that picking it changes nothing until a number moves.
         * In the rows' own resolution, whole Kelvin and a ten-thousandth
         * of tint, so that the fields read what was picked up rather than
         * 3099.99 and -0.00; the recorded temperature is whole already. */
        if (mode_of[sel] == VIEW_WB_MANUAL && lk->wb != VIEW_WB_MANUAL) {
            CRAWIlluminant cur;

            if (view_wb_applied(st, &cur)) {
                lk->manual.cct = roundf(cur.cct);
                lk->manual.tint = roundf(cur.tint * 1e4f) / 1e4f + 0.0f;
            }
        }
        lk->wb = mode_of[sel];
    }

    switch (lk->wb) {
    case VIEW_WB_AS_SHOT:
        if (st->dec.raw)
            view_note("Camera %.0f K, gains %.4f / 1 / %.4f",
                      st->prof.as_shot_cct, st->prof.as_shot_wb[0],
                      st->prof.as_shot_wb[2]);
        else if (st->dec.wb_kelvin > 0)
            view_note("Camera %d K", st->dec.wb_kelvin);
        else
            view_note("The camera recorded none: taken as the space's white");
        break;

    case VIEW_WB_AUTO: {
        const CRAWAwbEstimate *e = &st->awb.last;

        view_slider("Ambience", &st->awb.params.ambience, 0.0f, 1.0f, "%.2f");

        if (st->awb.have_illum)
            view_note("Tracking %.0f K, tint %+.4f",
                      st->awb.illum.cct, st->awb.illum.tint);
        else
            view_note("Nothing tracked yet, staying as shot");
        view_note("Estimate %s: %.0f K, tint %+.4f",
                  e->valid ? "good" : "bad", e->illuminant.cct, e->illuminant.tint);
        view_note("Confidence %.2f from %s, spread %.0f mired",
                  e->confidence, view_wb_source_name(e->source), e->spread_mired);
        break;
    }

    case VIEW_WB_MANUAL: {
        /* The spot white balance, a manual balance's tool and so a row of
         * Manual alone; leaving Manual with a pick armed calls it off. The
         * button arms it and the next click on the picture samples, so
         * while armed it is lit in the accent and the pointer over the
         * picture is a crosshair; a second press or Escape disarms. What a
         * pick finds goes into the two rows below, which is the readout;
         * what it could not find is the word beside the button, and
         * nothing else on the panel says anything about it. */
        bool lit = st->pick.armed;
        float cw = view_prop("Pick");
        bool pressed;

        if (lit)
            pressed = view_button_primary("Picking white###pick", ImVec2(cw * 0.5f, 0.0f));
        else
            pressed = ImGui::Button("Pick white###pick", ImVec2(cw * 0.5f, 0.0f));
        if (pressed)
            view_pick_arm(st, !st->pick.armed);
        if (st->pick.refused[0]) {
            ImGui::SameLine();
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(view_col_warn, "%s", st->pick.refused);
        }

        view_slider("Temperature", &lk->manual.cct, 2000.0f, 11500.0f, "%.0f K");
        view_slider("Tint", &lk->manual.tint, -0.2f, 0.2f, "%+.4f");
        break;
    }
    }

    if (lk->wb != VIEW_WB_MANUAL && st->pick.armed)
        view_pick_arm(st, false);

    if (!st->dec.raw &&
        (lk->wb != mode || lk->manual.cct != manual.cct ||
         lk->manual.tint != manual.tint || st->awb.params.ambience != ambience))
        st->sections.wb = true;
}

static void view_panel_exposure(struct view_state *st)
{
    struct view_look *lk = &st->look;
    static char labels[64][8];
    static const char *items[64];
    static int nb_items;
    int sel = 0;

    if (!nb_items) {
        nb_items = CRAW_MIN(view_nb_iso, 64);
        for (int i = 0; i < nb_items; i++) {
            snprintf(labels[i], sizeof(labels[i]), "%d", (int)view_iso_list[i]);
            items[i] = labels[i];
        }
    }

    /* The working index defaults to the as-shot ISO, which is what rates a
     * camera original at 1.0; anything else re-rates the clip. */
    for (int i = 0; i < nb_items; i++)
        if (fabsf(view_iso_list[i] - (lk->ei > 0.0f ? lk->ei : st->prof.as_shot_ei)) < 0.5f)
            sel = i;

    /* Re-rating a clip needs an as-shot index to rate it against; without
     * one - which is every developed clip this camera writes - only the
     * bias below means anything. */
    if (st->dec.raw || st->dec.meta_as_shot_ei > 0.0f) {
        int chosen = sel;

        if (view_combo_row("Exposure index", &chosen, items, nb_items, 12) &&
            chosen != sel) {
            lk->ei = view_iso_list[chosen];
            st->sections.exposure = true;
        }
    }

    /* A touched control switches the section on, RawTherapee's way: the
     * switch starts off and a lit one means the bias is in use. */
    if (view_slider("Bias", &lk->exposure, -3.0f, 3.0f, "%+.2f stops"))
        st->sections.exposure = true;

    /* The gain the picture really has, which a bypassed section takes back
     * to the as-shot rating while the two controls above keep their
     * numbers. */
    view_note("Gain %.2f, ceiling %.0f nits",
              view_total_gain(st) * exp2f(view_exposure_bias(st)),
              view_peak_ceiling(st));
}

/* What the record says the lens is doing, in one word: the sign of the
 * correction at the corner. Barrel distortion is corrected by pulling the
 * corners in, pincushion by pushing them out. */
static const char *view_lens_kind(const struct view_lens *ln)
{
    float g = ln->corner[1];

    return g < -5.0f ? "Barrel" : g > 5.0f ? "Pincushion" : "Near-neutral";
}

/* The "Lens" section: the geometric correction of DESIGN.md 16, driven by
 * the record every frame of a Panasonic ProRes RAW clip carries in its own
 * vendor metadata. The pipeline is always built with the pass (gpu.c), so
 * the switch is live: it reaches the next frame while playing and puts the
 * held picture through again while paused.
 *
 * The correction is the camera's, not the panel's. All this shows is what
 * the frame on screen brought with it, because a clip whose frames carry no
 * record cannot be corrected at all and the reason has to be visible rather
 * than inferred from nothing happening. On a clip that zooms while
 * recording the readout moves frame by frame, which is the record doing
 * exactly what it is for. The warp's own switch is the header's, greyed
 * there on a clip that carries no record; what is left here is the mode
 * and the readout. */
static void view_panel_lens(struct view_state *st)
{
    static const char *const frames[] = { "Fill", "Keep all" };
    struct view_lens *ln = &st->lens;
    bool have = ln->record.valid;
    int sel = ln->keep_sel;

    view_prop_value("Correction", !st->dec.raw ? "Not a raw clip"
                                : !have        ? "No record in this clip"
                                : ln->on       ? "On"
                                               : "Off");

    /* What the warp does with the frame, DESIGN.md 16.8: fill it and lose
     * the source's corners, which is the trade the camera's own crop makes,
     * or shrink the corrected picture until all of the source fits and
     * paint the rest black. On a clip with no record the row is greyed. */
    view_prop("Frame");
    ImGui::BeginDisabled(!have);
    view_segmented("lensframe", frames, 2, &sel);
    ImGui::EndDisabled();
    ln->keep_sel = sel;
    ln->keep = sel != 0;

    if (have) {
        /* Short enough for the panel's width: the corner shift per channel
         * and the chromatic difference are what a glance wants. */
        view_note("%s: corner %+.0f/%+.0f/%+.0f px RGB",
                  view_lens_kind(ln), (double)ln->corner[0],
                  (double)ln->corner[1], (double)ln->corner[2]);
        view_note("CA R-B %+.2f px, centre %.3f, %.3f",
                  (double)(ln->corner[0] - ln->corner[2]),
                  (double)ln->record.centre[0], (double)ln->record.centre[1]);

        /* The scale is the price of keeping the corners, so it is only
         * shown while the mode that charges it is on. It is the turned
         * frame's when the Rotate section turns it. */
        if (view_lens_keep_all(st)) {
            float scale = view_warp_scale(st);

            if (scale > 0.0f)
                view_note("Keeping all: scaled %.3f", (double)scale);
            else
                view_note("Keeping all: this record cannot be inverted");
        }
    }

    /* Two smoothed averages of libcraw's own pass time, the way the Denoise
     * section measures its own, so the cost shown is a measurement. */
    if (st->cfg.verbose) {
        if (ln->ms_on > 0.0 && ln->ms_off > 0.0)
            view_note("libcraw %.2f ms on, %.2f off: the warp costs %.2f ms",
                      ln->ms_on, ln->ms_off, ln->ms_on - ln->ms_off);
        else
            view_note("libcraw %.2f ms; toggle to measure", st->gpu.last_gpu_ms);
    }
}

/* The "Rotate" section: the turn of DESIGN.md 16.9, clockwise for a
 * positive angle. It is the lens warp's own resample, so it is offered for
 * raw clips, with or without a lens record, and the picture is magnified
 * just enough that the turned frame stays filled - using the corners the
 * lens correction would otherwise cut off before magnifying at all. The
 * readout under the slider is how much that is, the only thing the turn
 * costs; with the Lens section keeping everything it is the shrink
 * instead. */
static void view_panel_rotate(struct view_state *st)
{
    struct view_rotate *r = &st->rotate;
    float scale;

    /* A touched control switches the section on, RawTherapee's way. */
    if (view_slider("Angle", &r->degrees, -VIEW_ROTATE_MAX, VIEW_ROTATE_MAX,
                    "%+.2f\xc2\xb0"))
        r->on = true;

    if (view_rotation(st) == 0.0f)
        return;

    scale = view_warp_scale(st);
    if (!(scale > 0.0f))
        view_note("This record cannot be inverted");
    else if (view_lens_keep_all(st))
        view_note("Scaled %.3f to keep everything", (double)scale);
    else if (scale == 1.0f)
        view_note("Fills the frame without magnifying");
    else
        view_note("Magnified %.3f to fill the frame", (double)scale);
}

/* The "Sharpen" section: the capture sharpening of DESIGN.md 20, the last
 * stage on the camera RGB. The pipeline is always built with it (gpu.c), so
 * every row is live. The iteration count is the one a paused picture and an
 * export get; while the clip plays the pass runs at most
 * VIEW_SHARPEN_PLAY_ITERATIONS, and pausing puts the picture through again
 * at the full count. The contrast row greys on a camera with no noise
 * model, where there is nothing to measure contrast against and the stage
 * sharpens everything. */
static void view_panel_sharpen(struct view_state *st)
{
    struct view_sharpen *sh = &st->sharpen;
    bool grey = !view_denoise_model(st, NULL, NULL);

    view_slider("Radius", &sh->radius, 0.3f, CRAW_SHARPEN_MAX_RADIUS, "%.3f px");
    view_slider_int("Iterations", &sh->iterations, 1, CRAW_SHARPEN_MAX_ITERATIONS, "%d");

    ImGui::BeginDisabled(grey);
    view_slider("Contrast", &sh->contrast, 0.0f, 4.0f, "%.2f");
    ImGui::EndDisabled();
}

/* The "Denoise" section: the raw domain stage of DESIGN.md 14, which runs
 * on the CFA planes before the debayer. The pipeline is always built with
 * the pass (gpu.c), so everything here is live: the strength reaches the
 * next frame and a change while paused puts the picture through again.
 *
 * The model is the camera's, not the panel's. All this shows is which one
 * was found for the frame's own ISO, because a clip the camera has not been
 * characterized at cannot be denoised at all and the reason has to be
 * visible rather than inferred from nothing happening. The stage's own
 * switch is the header's; what is left here is the strength, what the
 * frame on screen resolved, the temporal average in front of the stage
 * and the chroma stage after the debayer. */
static void view_panel_denoise(struct view_state *st)
{
    struct view_denoise *d = &st->denoise;

    view_slider("Strength", &d->strength, 0.0f, 3.0f, "%.2f");

    /* 1.0 is "remove noise at exactly the modelled level", so the numbers
     * either side of it are worth naming. Zero leaves the spatial filter
     * out, for the temporal average or the chroma stage alone. */
    view_note("%s", d->strength <= 0.0f ? "Spatial filter off"
                  : d->strength < 0.95f ? "Under the modelled level"
                  : d->strength < 1.05f ? "At the modelled level"
                  : d->strength < 2.05f ? "Smooths across real detail"
                                        : "Heavy");

    /* The temporal average in front of the raw stage, DESIGN.md 14.7: how
     * much of the previous frame a sample keeps where the two agree within
     * the model's noise. It needs the model, like the strength above. The
     * row stops short of 1, at which a static sample would keep its first
     * frame for ever. */
    view_slider("Temporal", &d->temporal, 0.0f, 0.95f, "%.2f");

    /* The chroma denoise after the debayer, DESIGN.md 14.4: the quarter
     * resolution chroma band replaced by its luma-guided smoothing. It
     * needs no model, so it runs whenever the section is on and this is
     * above zero. */
    view_slider("Chroma", &d->chroma, 0.0f, 1.0f, "%.3f");

    if (!d->have_model)
        view_note("No noise model for ISO %.0f: the raw stage is off", (double)d->iso);
    else
        view_note("Noise a %.3g, b %.2g at ISO %.0f%s",
                  (double)d->model.a, (double)d->model.b, (double)d->iso,
                  d->on ? ", before the debayer" : "");

    /* Two smoothed averages of libcraw's own pass time, one for the frames
     * that ran the stage and one for the frames that did not, so the cost
     * shown is a measurement and not an estimate. It only appears once the
     * clip has been played both ways. */
    if (st->cfg.verbose) {
        if (d->ms_on > 0.0 && d->ms_off > 0.0)
            view_note("libcraw %.2f ms on, %.2f off: the pass costs %.2f ms",
                      d->ms_on, d->ms_off, d->ms_on - d->ms_off);
        else
            view_note("libcraw %.2f ms; toggle to measure", st->gpu.last_gpu_ms);
    }
}

/* A checkbox in the label column and what it means beside it. */
static bool view_check_row(const char *label, bool *v, const char *on,
                           const char *off)
{
    float x = ImGui::GetCursorPosX();
    float w = ImGui::GetContentRegionAvail().x;
    bool changed = ImGui::Checkbox(label, v);

    ImGui::SameLine(x + floorf(w * VIEW_LABEL_FRAC));
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
    ImGui::TextUnformatted(*v ? on : off);
    ImGui::PopStyleColor();

    return changed;
}

/* The "Adjustments" section: soft light and vibrance, the two RawTherapee
 * tools adjust.c reimplements. It sits between Look and Film grain because
 * that is where the hooks sit in the render: after the film simulation the
 * Look section chooses, before the grain. The header above it is the
 * section's switch: off leaves both hooks unattached at the amounts they
 * hold. */
static void view_panel_adjust(struct view_state *st)
{
    struct view_adjust *a = &st->adjustments;
    char buf[128];
    bool grey, touched = false;

    grey = view_grey_begin(st,
        "Not applied: the export is scene referred, and these two are "
        "display-referred operators on values in [0,1].");

    /* Soft light first, even though it runs last: it is the one control
     * most people came for, and the panel is not the pipeline. A touched
     * control switches the section on, RawTherapee's way: the switch
     * starts off and a lit one means an adjustment is in use. */
    touched |= view_slider("Soft light", &a->softlight, 0.0f, 100.0f, "%.1f");
    touched |= view_slider("Pastels", &a->pastels, -100.0f, 100.0f, "%+.1f");
    touched |= view_slider("Saturated", &a->saturated, -100.0f, 100.0f, "%+.1f");
    touched |= view_slider("Threshold", &a->threshold, 0.0f, 100.0f, "%.1f");

    touched |= view_check_row("Protect skin", &a->protect_skin,
                              "Warm hues held back", "Every hue alike");
    touched |= view_check_row("Avoid shift", &a->avoid_shift,
                              "Chroma kept in gamut", "Chroma clipped");

    view_note("%s", view_adjust_summary(a, buf, sizeof(buf)));

    if (touched)
        st->sections.adjust = true;

    view_grey_end(grey);
}

/* The body of the "Film grain" section, under the header whose switch is
 * grain.on. Grain is applied after colour management, on display-encoded
 * values, so a scene-referred export carries none of it and neither does
 * the screen while one is selected. */
static void view_panel_grain(struct view_state *st)
{
    struct view_grain *g = &st->grain;
    static const char *responses[CRAW_GRAIN_RESPONSE_NB] = {
        "Negative", "Print", "Reversal", "Flat",
    };
    static const char *gauges[CRAW_GRAIN_GAUGE_NB];
    static const char *stocks[CRAW_GRAIN_STOCK_NB];
    static bool names_done;
    char buf[128];
    bool grey;

    if (!names_done) {
        for (int i = 0; i < CRAW_GRAIN_GAUGE_NB; i++)
            gauges[i] = craw_grain_gauge_name((enum craw_grain_gauge)i);
        for (int i = 0; i < CRAW_GRAIN_STOCK_NB; i++)
            stocks[i] = craw_grain_stock_name((enum craw_grain_stock)i);
        names_done = true;
    }

    grey = view_grey_begin(st, "Not applied: the export is scene referred, and "
                               "grain is a display-referred operator.");

    view_combo_row("Gauge", &g->gauge, gauges, CRAW_GRAIN_GAUGE_NB);
    view_combo_row("Stock", &g->stock, stocks, CRAW_GRAIN_STOCK_NB, 16);
    view_combo_row("Response", &g->response, responses, CRAW_GRAIN_RESPONSE_NB);
    view_slider("Amount", &g->amount, 0.0f, 3.0f, "%.2f");
    view_slider("Size", &g->size, 0.25f, 4.0f, "%.2f");
    view_check_row("Boil", &g->boil, "New grain each frame", "Held still");

    /* What the model made of the settings - whether there is any grain at
     * all is the header's switch - and chroma, the one setting with a
     * meaningful "unset": below zero the stock's own figure is used, which
     * is what most people want. */
    if (g->chroma < 0.0f)
        view_note("%s, chroma from the stock", view_grain_summary(g, buf, sizeof(buf)));
    else
        view_note("%s, chroma %.2f", view_grain_summary(g, buf, sizeof(buf)),
                  (double)g->chroma);

    /* Where the numbers came from, since half the stocks are estimates. */
    view_note_wrapped(craw_grain_stock_source((enum craw_grain_stock)g->stock),
                      view_col_text_dim);

    view_grey_end(grey);
}

/* The sound rows of the Video block: a combo of the six modes, with the
 * ones the file's container will not carry greyed and labelled with the
 * muxer's own reason, and under it the one number the mode takes. What the
 * container is comes from the file field: the popup greys opus and flac
 * under a `.mov`.
 *
 * A slowed clip is exported without sound (DESIGN.md 23.3), so while it is
 * both rows are greyed and the note under them says that instead of what
 * the mode does. The mode is left as it was chosen, on the combo where it
 * can be seen, and it is what the next export at the clip's own rate
 * writes. */
static void view_panel_export_audio(struct view_state *st, const char *path)
{
    struct view_preset_sel *sel = &st->look.video_preset;
    const struct view_audio_spec *m;
    char label[160];
    int mode = CRAW_CLAMP(sel->audio_mode, 0, VIEW_NB_AUDIO_MODES - 1);
    bool slowed = view_slowed(st);

    ImGui::BeginDisabled(slowed);

    m = &view_audio_modes[mode];
    if (m->field)
        snprintf(label, sizeof(label), "%s %d %s", m->name, sel->audio_q[mode],
                 m->field);
    else
        snprintf(label, sizeof(label), "%s", m->name);

    /* The popup grows past the combo's own width for a refusal: "mov: opus
     * only supported in MP4" is what it has to say. */
    view_prop("Audio");
    if (ImGui::BeginCombo("##audio", label)) {
        for (int i = 0; i < VIEW_NB_AUDIO_MODES; i++) {
            const char *no = view_encode_audio_why_not(st, path, i);

            if (no) {
                snprintf(label, sizeof(label), "%s: %s",
                         view_audio_modes[i].name, no);
                ImGui::BeginDisabled();
                ImGui::Selectable(label);
                ImGui::EndDisabled();
            } else if (ImGui::Selectable(view_audio_modes[i].name, i == mode)) {
                mode = i;
            }
        }
        ImGui::EndCombo();
    }
    sel->audio_mode = mode;
    m = &view_audio_modes[mode];

    /* The number the mode takes, or what the mode is instead of one. */
    if (m->field) {
        int step = !strcmp(m->field, "level") ? 1 : 16;
        char name[48];

        snprintf(name, sizeof(name), "%c%s", toupper((unsigned char)m->field[0]),
                 m->field + 1);
        view_number_row(name, &sel->audio_q[mode], m->min, m->max,
                        (float)step * 0.2f, "%d");
    }
    if (slowed)
        view_note("No sound at this rate");
    else if (mode == VIEW_AUDIO_OPUS)
        view_note("120 ms frames");
    else if (mode == VIEW_AUDIO_COPY)
        view_note("The clip's own track");
    else if (mode == VIEW_AUDIO_WAV)
        view_note("PCM at the clip's depth");
    else if (!m->field)
        view_note("No sound");

    ImGui::EndDisabled();
}

/* ------------------------------------------------- the export file fields */

/* A path's extension - the last dot after the last slash, dot included -
 * and where it starts; "" and the whole path when there is none. */
static const char *view_field_ext(const char *path, size_t *stem)
{
    const char *slash = strrchr(path, '/');
    const char *dot = strrchr(slash ? slash + 1 : path, '.');

    *stem = dot ? (size_t)(dot - path) : strlen(path);
    return dot ? dot : "";
}

/* The file field of an export section follows its default, `want` - the
 * clip's name and the frame or the range on screen, in ~/Processed, under
 * the section's own extension - except for what was typed into it, which
 * is told from what was put there by `def`, the default the field was
 * last given. The two parts of a name are kept differently. The
 * extension, once typed, is the user's format and stays: on this clip,
 * the next and every one after, until it is typed back. The name before
 * it, once typed, is kept on the clip it was typed on and on no other:
 * the next clip gets its own name again, under the user's extension. The
 * name used to be kept across clips as well, so a name typed for one
 * clip was the next one's too, and exporting the next wrote over the
 * first (Lynne). A name typed without an extension gets the default's
 * when the button is pressed and is not a choice of extension.
 *
 * `clip` is the clip the field was last given a default on, which is how
 * the next one is told. Called while the field is not being typed into. */
static void view_field_follow(const struct view_state *st, char *field,
                              size_t size, char *def, size_t def_size,
                              int *clip, const char *want)
{
    char out[VIEW_MAX_PATH];
    const char *fext = "", *dext, *wext;
    size_t fstem = 0, dstem, wstem;
    bool stem_typed = false, ext_typed = false;

    if (field[0]) {
        fext = view_field_ext(field, &fstem);
        dext = view_field_ext(def, &dstem);
        stem_typed = fstem != dstem || strncmp(field, def, fstem) != 0;
        ext_typed = fext[0] && strcmp(fext, dext) != 0;
    }
    wext = view_field_ext(want, &wstem);

    if (stem_typed && *clip == st->clip)
        /* The user's name on the user's clip: left alone. */
        snprintf(out, sizeof(out), "%s", field);
    else
        snprintf(out, sizeof(out), "%.*s%s", (int)wstem, want,
                 ext_typed ? fext : wext);

    snprintf(field, size, "%s", out);
    snprintf(def, def_size, "%s", want);
    *clip = st->clip;
}

/* A file field across the row under its label, with what the extension
 * decides said in the label, and whether it is being typed into. */
static bool view_file_field(const char *label, const char *id, char *field,
                            size_t size)
{
    ImGui::PushFont(NULL, VIEW_FONT_SMALL_PX);
    ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputText(id, field, size);

    return ImGui::IsItemActive();
}

/* The Export video section: the codec preset and the colour
 * space cascades, the pixel format, the size, whatever knobs the chosen
 * family declares, the sound, the threads, a frame range, the file, one
 * button, a progress bar and a way to stop. The selection itself lives in
 * the look, so it survives the export and carries from clip to clip; the
 * file field follows its default until it is typed into, and then it is
 * the user's and carries too. */
static void view_panel_export_video(struct view_state *st)
{
    struct view_video *v = &st->video;
    struct view_preset_sel *sel = &st->look.video_preset;
    const struct view_preset_family *fam;
    char want[VIEW_MAX_PATH], label[128];
    const char *codec, *why;
    bool busy = v->active;
    bool scene, vulkan;
    view_row r;
    /* The last frame there is, which is the most either field can hold:
     * a range cannot reach past the clip. A clip whose length FFmpeg
     * cannot tell has no such bound, and the engine finds the end by
     * running into it. */
    int last = st->dec.nb_frames > 0
             ? (int)CRAW_MIN(st->dec.nb_frames - 1, INT_MAX - 1) : INT_MAX - 1;

    /* The whole clip until a range has been set: Ctrl+1 and Ctrl+2 mark
     * the start and the end at the frame on screen, and the two fields
     * below take numbers. A range set on a longer clip is cut to this one
     * without a word: the fields' own bounds are what say how far it can
     * go. */
    if (!busy) {
        if (!v->range_set) {
            v->start_ui = 0;
            v->end_ui = last;
        } else {
            v->start_ui = CRAW_CLAMP(v->start_ui, 0, last);
            v->end_ui = CRAW_CLAMP(v->end_ui, v->start_ui, last);
        }
    }

    if (sel->family < 0 || sel->family >= view_nb_preset_families)
        view_presets_default(sel);

    fam = &view_preset_families[sel->family];
    vulkan = view_presets_family_is_vulkan(sel->family);

    /* --codec still wins, for an encoder that is not one of the ten. */
    codec = st->cfg.export_codec;
    if (!codec || !codec[0])
        codec = fam->encoder;

    /* The file field follows its default - the clip's name, the range and
     * the family's container, in ~/Processed - the way view_field_follow()
     * says: a typed extension stays, a typed name stays on its clip. Not
     * while a run is on: the name the run is writing is the one to show.
     * And not while the field is being typed into: a field cleared to type
     * a fresh name into would otherwise be refilled under the cursor. */
    if (!busy && !v->field_active) {
        view_encode_default_path(st, v->start_ui, v->end_ui, sel, codec,
                                 want, sizeof(want));
        view_field_follow(st, v->field, sizeof(v->field), v->field_def,
                          sizeof(v->field_def), &v->field_clip, want);
    }

    view_menu_row(st, VIEW_MENU_PRESET, "Codec",
                  view_presets_label(sel, label, sizeof(label)));
    view_menu_row(st, VIEW_MENU_COLOUR, "Colour",
                  view_colours[CRAW_CLAMP(sel->colour, 0, VIEW_NB_COLOURS - 1)].name);

    scene = view_colours[CRAW_CLAMP(sel->colour, 0, VIEW_NB_COLOURS - 1)].scene;

    /* The pixel format the picture is written in. The list is what the
     * encoder was found to accept - nothing else is offered - and the entry
     * in use is the preset's own pin until the row is touched, after which
     * the choice outlives a preset change and dies with a family one. */
    {
        /* A scene-referred export is written in one of the family's float
         * formats whatever the row held before, and ACEScct in ten bits or
         * more: say which. */
        enum AVPixelFormat f = view_presets_sel_out_fmt(sel);

        if (f == AV_PIX_FMT_NONE)
            f = view_presets_sel_fmt(sel);

        /* The format is known from the table the moment the preset is,
         * while the *list* has to be asked of the encoder one format at a
         * time; so the row says what the file will be written as straight
         * away and the menu under it fills in behind. A preset that pins
         * nothing leaves the choice to the engine until the row is used. */
        if (f != AV_PIX_FMT_NONE)
            view_presets_fmt_label(f, label, sizeof(label));
        else
            snprintf(label, sizeof(label), "chosen by the engine");
        view_menu_row(st, VIEW_MENU_PIXFMT, "Pixel format", label);
    }

    /* Why the selection cannot be used, said once and in full rather than
     * left for the encoder to discover. */
    why = view_presets_preset_why_not(sel->family, sel->preset);
    if (!why)
        why = view_presets_sel_colour_why_not(sel, sel->colour);
    /* A format the row still holds from a preset whose profile allowed it,
     * under one whose profile does not: the menu greys such an entry, but a
     * preset change can walk the selection into one all the same, and the
     * profile is the thing that has to win. */
    if (!why && !view_colours[sel->colour].needs_deep_rgb)
        why = view_presets_fmt_why_not(sel->family, sel->preset,
                                       view_presets_sel_out_fmt(sel));
    /* The size row and a scene-referred space cannot both have their way:
     * the direct path has nothing to resample with, and inventing a scaler
     * for linear light is a decision nobody asked this program to make. */
    if (!why && scene && sel->size != VIEW_SIZE_FULL)
        why = "a scene-referred export is written straight from the "
              "pipeline, at the source's own size; set the size row to full";
    if (!why && scene && !st->dec.raw)
        why = "a scene-referred export comes out of the raw pipeline, and "
              "this clip is already developed";
    /* A hardware engine's largest picture, against what the size row
     * makes of the crop: H.264 on RADV stops at 4096, and a 5.8K clip at
     * full size is past it. */
    if (!why && !scene) {
        int x, y, cw, ch, w, h;

        view_crop_disp_rect(st, &x, &y, &cw, &ch);
        view_presets_size(sel, cw, ch, &w, &h);
        why = view_presets_extent_why_not(avcodec_find_encoder_by_name(codec),
                                          w, h);
    }
    /* What the file's name decides - the container, whether it carries the
     * encoder and the sound - is judged when the button is pressed, by
     * view_encode_begin(), and never while the name is being typed: every
     * name on the way from `.mkv` to `.mp4` is a name that picks no muxer,
     * and a row that appeared above the field for as long as that lasted
     * pushed the field down the panel and out from under the cursor. A row
     * that comes and goes under the cursor is not a refusal, it is a
     * fight. */
    if (why) {
        snprintf(want, sizeof(want), "Cannot export: %s.", why);
        view_note_wrapped(want, view_col_warn);
    } else if (scene) {
        view_note_wrapped(view_colours[sel->colour].log
                          ? "Scene referred: written straight from the pipeline "
                            "through the ACEScct curve, with no tone curve, no "
                            "gamut mapping, no film simulation and no grain."
                          : "Scene referred: written straight from the pipeline "
                            "as float, with no tone curve, no gamut mapping, no "
                            "film simulation and no grain.",
                          view_col_text_dim);
    }

    /* The size, which every export has and the direct one refuses. The
     * rectangle is the rotated one, because the file is written upright:
     * what the row halves and what the readout says are the picture as it
     * comes out, not the readout behind it. */
    {
        int x, y, cw, ch;

        view_crop_disp_rect(st, &x, &y, &cw, &ch);
        view_combo_row("Size", &sel->size, view_size_names, VIEW_NB_SIZES);

        if (sel->size == VIEW_SIZE_TYPED) {
            float w = view_prop("Width \xc3\x97 height");

            r = view_row_begin(2);
            r.w = w;
            view_cell(r, 0);
            view_number("##width", &sel->size_w, 2, 16384, 1.0f, "%d");
            view_cell(r, 1);
            view_number("##height", &sel->size_h, 2, 16384, 1.0f, "%d");
        }

        view_note("Writes %s", view_presets_size_label(sel, cw, ch, label, sizeof(label)));
    }

    /* The target peak of a PQ export, which is part of what the target is.
     * How the picture is mapped into it is not a row here any more: a
     * display-referred export is mapped exactly as the window is, with
     * libplacebo's own defaults, so what the file holds is what is on
     * screen. */
    if (view_colours[sel->colour].hdr && sel->colour == VIEW_COLOUR_2020_PQ)
        view_number_row("Peak", &sel->hdr_peak, 100, 10000, 5.0f, "%d nits");

    /* Only the rows the family declares. */
    if (fam->ctrl & VIEW_CTRL_ASYNC)
        view_number_row("Async depth", &sel->async, fam->async_min, fam->async_max,
                        0.05f, "%d");
    if (fam->ctrl & VIEW_CTRL_QP)
        view_number_row("QP", &sel->qp, fam->qp_min, fam->qp_max, 0.2f, "%d");
    if (fam->ctrl & VIEW_CTRL_CPU)
        view_number_row("CPU preset", &sel->cpu, fam->cpu_min, fam->cpu_max, 0.05f, "%d");
    if (fam->ctrl & VIEW_CTRL_RC && fam->nb_rc > 0) {
        const char *labels[8];
        const struct view_rc_spec *rc;
        int n = CRAW_MIN(fam->nb_rc, 8), rcsel = CRAW_CLAMP(sel->rc, 0, n - 1);

        for (int i = 0; i < n; i++)
            labels[i] = fam->rc[i].label;
        view_prop("Rate control");
        if (view_segmented("rc", labels, n, &rcsel))
            sel->rc = rcsel;

        rc = &fam->rc[CRAW_CLAMP(sel->rc, 0, fam->nb_rc - 1)];
        if (rc->bitrate) {
            int mbit = CRAW_MAX(sel->kbit / 1000, 1);

            /* Mbit/s on the panel, kbit/s underneath: a bitrate field that
             * counted in kilobits would need five digits to say anything
             * useful about a 5.8K export. */
            view_number_row("Bitrate", &mbit, 1, 2000, 0.5f, "%d Mbit/s");
            sel->kbit = mbit * 1000;
        } else {
            char name[48];

            snprintf(name, sizeof(name), "%s", rc->label);
            view_number_row(name, &sel->quant, rc->qmin, rc->qmax, 0.2f, "%d");
        }
    }

    /* The threads, which only a software encoder has: greyed under a
     * Vulkan family, with the note saying why. */
    ImGui::BeginDisabled(vulkan);
    view_number_row("Threads", &sel->threads, 0, view_presets_max_threads(), 0.2f,
                    vulkan ? "none" : sel->threads ? "%d" : "auto");
    ImGui::EndDisabled();
    if (vulkan)
        view_note("A Vulkan encoder has no threads to set");

    view_panel_export_audio(st, v->field);

    {
        float w = view_prop("Frames");

        r = view_row_begin(2);
        r.w = w;
        view_cell(r, 0);
        if (view_number("##start", &v->start_ui, 0, last, 1.0f, "%d"))
            v->range_set = true;
        view_cell(r, 1);
        if (view_number("##end", &v->end_ui, v->start_ui, last, 1.0f, "to %d"))
            v->range_set = true;
    }

    v->field_active = view_file_field("File, its extension picks the container",
                                      "##videofile", v->field, sizeof(v->field));

    /* The one stretch of a locked panel that stays live: the stop button,
     * the progress bar and the status under them. Everything else is
     * greyed from the top of the panel down while the run is on. */
    if (busy)
        ImGui::BeginDisabledOverrideReenable();

    {
        float cw;
        bool pressed;

        r = view_row_begin(2);
        cw = view_cell(r, 0);
        if (busy)
            pressed = view_button_danger("Stop", ImVec2(cw, 0.0f));
        else
            pressed = view_button_primary("Export video", ImVec2(cw, 0.0f));
        if (pressed) {
            if (busy) {
                view_encode_cancel(st);
            } else if (!why) {
                struct view_encode_opts o = {};

                o.start = v->start_ui;
                o.end   = v->end_ui;
                o.path  = v->field;
                o.sel   = sel;
                o.codec = codec;
                o.enc   = st->cfg.export_enc;

                /* A name without an extension gets the family's container,
                 * in the field itself, so that what the panel shows is what
                 * is written. Everything else about the name - a container
                 * that will not carry the encoder or the sound - is refused
                 * by the engine, on the status line under this button. */
                view_encode_complete_path(sel, codec, v->field, sizeof(v->field));
                v->range_set = true;
                view_encode_begin(st, &o);
            }
        }
        cw = view_cell(r, 1);
        ImGui::ProgressBar((float)v->progress, ImVec2(cw, 0.0f), "");
    }

    if (v->status[0])
        view_note_wrapped(v->status, view_col_text_dim);
    if (v->path[0]) {
        const char *name = strrchr(v->path, '/');

        view_note("%.46s", name ? name + 1 : v->path);
    }

    if (busy)
        ImGui::EndDisabledOverrideReenable();
}

/* The Export image section: the frame on screen, one image, the format
 * picked by the file field's extension. The field follows the frame on
 * screen, a .jpg under the clip's name, the way view_field_follow() says:
 * a typed extension stays, a typed name stays on its clip. */
static void view_panel_export_still(struct view_state *st)
{
    struct view_export *e = &st->exp;
    char want[VIEW_MAX_PATH];
    int x, y, w, h;
    view_row r;
    float cw;

    /* Not while the field is being typed into: a field cleared to type a
     * fresh name into would otherwise be refilled under the cursor. */
    if (!e->field_active) {
        view_export_default_path(st, st->index, want, sizeof(want));
        view_field_follow(st, e->field, sizeof(e->field), e->field_def,
                          sizeof(e->field_def), &e->field_clip, want);
    }

    e->field_active = view_file_field("File, its extension picks the format",
                                      "##stillfile", e->field, sizeof(e->field));

    r = view_row_begin(2);
    cw = view_cell(r, 0);
    if (view_button_primary("Export image", ImVec2(cw, 0.0f)) && !e->active) {
        /* The current frame, and nothing but it: a range of stills is a
         * video encode, which this is not. The encoder is the one the
         * name implies; there is no codec box, the extension is the
         * choice. */
        st->playing = false;
        /* A name without an extension gets the .jpg the default has, in
         * the field itself, so that what the panel shows is what is
         * written. */
        view_export_complete_path(e->field, sizeof(e->field));
        view_export_begin(st, st->index, e->field, NULL);
    }
    /* What the file will be, which is the rotated size: the still is
     * written upright like everything else. */
    view_cell(r, 1);
    ImGui::AlignTextToFramePadding();
    ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
    if (view_crop_disp_rect(st, &x, &y, &w, &h))
        ImGui::Text("%d \xc3\x97 %d, cropped", w, h);
    else
        ImGui::Text("%d \xc3\x97 %d", w, h);
    ImGui::PopStyleColor();

    if (e->status[0])
        view_note_wrapped(e->status, view_col_text_dim);
}

/* The "Crop" section: what a left drag has drawn. The header's switch is
 * lit for a crop in use and for the mode the drag draws in, which `c` is
 * the other hand on, and starting to draw is what clears a rectangle - so
 * there is no clear button, only the readout. The rectangle is held in
 * source pixels and reported on the picture as it is shown, so the readout
 * says which region the export will write whatever the window is doing
 * with it. On a rotated clip those are two different rectangles and the
 * one worth printing is the one that was drawn - `--crop` takes the
 * source's, which is that one turned back. */
static void view_panel_crop(struct view_state *st)
{
    int x, y, w, h;
    char buf[96];

    /* "(clip)" while the rectangle is the one the clip's own frame header
     * marked rather than one that was drawn or asked for: it says why a
     * picture that was never cropped by hand exports smaller than its
     * readout, and that the rectangle is the picture on screen rather than
     * a shade over it. */
    if (view_crop_disp_rect(st, &x, &y, &w, &h)) {
        snprintf(buf, sizeof(buf), "%d \xc3\x97 %d at %d, %d%s", w, h, x, y,
                 st->crop.from_clip ? " (the clip's)" : "");
        view_prop_value("Rectangle", buf);
    } else {
        view_prop_value("Rectangle", "None");
    }

    if (st->crop.on)
        view_note("Drag over the picture to draw; c or the switch keeps it");
    else if (view_crop_rect(st, &x, &y, &w, &h))
        view_note("The switch drops it; c draws a new one");
    else
        view_note("c or the switch draws one");
}

/* The "Clip" block: what the file says about the recording, as one label
 * and one value per row, and only the rows the file has something for. It
 * is information and nothing else - nothing here is clicked - so it is set
 * in the readouts' size, the labels dimmed and the values carrying the
 * row. A value too long for its column is cut with an ellipsis rather
 * than let run under the scrollbar: the panel's width is fixed and a
 * camera's model name is not.
 *
 * It sits under everything live - the readouts, the scrub bar, the mute
 * and the volume - and above the first section, rather than straight
 * under the clip line, because its height is the one thing in the panel's
 * top that changes with the clip: two rows for a phone's MP4, eleven for
 * a camera's HEVC take. Put above the scrub bar it would move the bar under
 * the pointer at every clip switch; here the live rows keep their places
 * and the block reads as the heading of the sections that follow. */
static void view_panel_clip(struct view_state *st)
{
    struct view_info_row rows[VIEW_INFO_MAX];
    int n = view_decoder_info(&st->dec, rows, VIEW_INFO_MAX);
    float x = ImGui::GetCursorPosX();
    float lw = floorf(ImGui::GetContentRegionAvail().x * VIEW_LABEL_FRAC);

    if (!n)
        return;

    ImGui::Dummy(ImVec2(0.0f, 2.0f));
    ImGui::PushFont(NULL, VIEW_FONT_SMALL_PX);
    for (int i = 0; i < n; i++) {
        char label[64], value[96];

        /* The camera is in the log's clip line and the lens row says
         * which system it is; on the panel it was a row saying the same
         * thing every time (Lynne). */
        if (!strcmp(rows[i].label, "camera"))
            continue;

        snprintf(label, sizeof(label), "%c%s", toupper((unsigned char)rows[i].label[0]),
                 rows[i].label + 1);
        ImGui::PushStyleColor(ImGuiCol_Text, view_col_text_dim);
        ImGui::TextUnformatted(label);
        ImGui::PopStyleColor();
        ImGui::SameLine(x + lw);
        view_fit_text(ImGui::GetContentRegionAvail().x - 2.0f, rows[i].value,
                      value, sizeof(value));
        ImGui::TextUnformatted(value);
    }
    ImGui::PopFont();
    ImGui::Dummy(ImVec2(0.0f, 4.0f));
}

/* The bottom of the panel, while files are marked for deletion
 * (Shift+Alt+Delete, struct view_bin), down at its bottom edge while the
 * sections above leave room and after the last of them when they do not:
 * the button that deletes them, and beside it whether the file on screen
 * is one of them - with its JPEG when a still has one. Under them, what
 * went wrong with the last press, if anything did. Nothing at all while
 * nothing is marked. */
static void view_panel_bin(struct view_state *st)
{
    const struct view_bin *bin = &st->bin;
    const ImGuiStyle &style = ImGui::GetStyle();
    const struct view_bin_entry *b;
    char label[48];

    if (!bin->entries || (!bin->nb_marked && !bin->status[0]))
        return;
    b = &bin->entries[st->clip];

    /* Pushed down by the room left under the sections, short of the
     * bottom by a few pixels so that the panel never comes out a pixel
     * taller than its window and scrolls for nothing. */
    {
        float need = 1.0f + 2.0f * style.ItemSpacing.y, gap;

        if (bin->nb_marked)
            need += ImGui::GetFrameHeightWithSpacing();
        if (bin->status[0]) {
            ImGui::PushFont(NULL, VIEW_FONT_SMALL_PX);
            need += ImGui::CalcTextSize(bin->status, NULL, false,
                                        ImGui::GetContentRegionAvail().x).y +
                    style.ItemSpacing.y;
            ImGui::PopFont();
        }

        gap = ImGui::GetContentRegionAvail().y - need - 4.0f;
        if (gap >= 1.0f)
            ImGui::Dummy(ImVec2(0.0f, gap));
    }

    ImGui::Separator();

    if (bin->nb_marked) {
        view_row r = view_row_begin(2);
        float cw;

        snprintf(label, sizeof(label), "Delete %d marked", bin->nb_marked);
        cw = view_cell(r, 0);
        if (view_button_danger(label, ImVec2(cw, 0.0f)))
            st->bin.go = true;
        if (b->marked) {
            view_cell(r, 1);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(view_col_danger, "%s",
                               b->jpg ? "This one, and its JPEG" : "This one");
        }
    }

    if (bin->status[0])
        view_note_wrapped(bin->status, view_col_warn);
}

/* An algorithm's name as a button says it: an acronym in capitals, a word
 * with its first letter up. */
static void view_title_case(const char *in, char *out, size_t n)
{
    snprintf(out, n, "%s", in);
    if (strlen(out) <= 3)
        for (char *p = out; *p; p++)
            *p = (char)toupper((unsigned char)*p);
    else
        out[0] = (char)toupper((unsigned char)out[0]);
}

/* The top of the panel: the clip's name in the bold face, which of how
 * many when there is more than one, and under it in the readouts' size
 * what it is and how it is playing. The rotation joins the codec when the
 * clip carries one, since it is the other thing about the clip that
 * decides what is on screen and it is otherwise invisible - a portrait
 * picture looks like a portrait clip. */
static void view_panel_title(struct view_state *st)
{
    const char *file = strrchr(st->cfg.input, '/');
    const char *codec = st->dec.codec_name ? st->dec.codec_name : "?";
    const char *kind = st->dec.raw     ? "raw"
                     : st->dec.hwaccel ? "video, decoded on the GPU"
                                       : "video, decoded on the CPU";
    float x = ImGui::GetCursorPosX(), w = ImGui::GetContentRegionAvail().x;
    char rot[24] = "", count[32] = "", name[160];

    file = file ? file + 1 : st->cfg.input;
    if (st->dec.rotation)
        snprintf(rot, sizeof(rot), ", %d\xc2\xb0", st->dec.rotation_deg);
    if (st->cfg.nb_inputs > 1)
        snprintf(count, sizeof(count), "%d of %d", st->clip + 1, st->cfg.nb_inputs);

    ImGui::PushFont(view_font_strong, 0.0f);
    view_fit_text(w - view_text_w(count) - 12.0f, file, name, sizeof(name));
    ImGui::TextUnformatted(name);
    ImGui::PopFont();
    if (count[0]) {
        ImGui::SameLine(x + w - view_text_w(count));
        ImGui::TextColored(view_col_text_dim, "%s", count);
    }

    view_note("%s %s%s", codec, kind, rot);
    /* Whatever the playlist has to report: a file that would not open, a
     * switch refused. Under the clip line, because that is what it is
     * about, and never behind a fold. */
    if (st->clip_status[0])
        view_note_wrapped(st->clip_status, view_col_warn);
    view_note("Frame %" PRId64 " of %" PRId64 " \xc2\xb7 %.3f s",
              st->index, st->dec.nb_frames ? st->dec.nb_frames - 1 : 0, st->pts);
    view_note("%.0f/%.0f fps \xc2\xb7 decode %.1f ms \xc2\xb7 libcraw %.2f ms",
              st->proc_fps, st->fps, st->decode_ms, st->gpu.last_gpu_ms);
}

static void view_panel(struct view_state *st)
{
    struct view_look *lk = &st->look;
    bool locked = view_ui_locked(st);

    /* A running video export locks the panel from here to the end: every
     * row is drawn and none takes input, by mouse, wheel or key. The engine
     * owns the decoder - it seeks it and steps it - and has copied the
     * settings, so a slider that moved would be changing a picture nobody
     * is looking at and nothing about the file. The Video block's stop
     * button, progress bar and status are the one stretch left live, which
     * view_panel_export_video() re-enables for exactly those three. */
    ImGui::BeginDisabled(locked);

    view_panel_title(st);
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    view_panel_transport(st);

    if (st->audio.available) {
        float x = ImGui::GetCursorPosX(), w = ImGui::GetContentRegionAvail().x;
        float pct = st->volume * 100.0f, lo = 0.0f, hi = 100.0f;

        ImGui::Checkbox("Mute", &st->mute);
        ImGui::SameLine(x + floorf(w * VIEW_LABEL_FRAC));
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (view_slider_scalar("##volume", ImGuiDataType_Float, &pct, &lo, &hi,
                               "%.0f%%", 0))
            st->volume = pct / 100.0f;

        view_note("Sound: %s", st->audio.mapping);
        view_note("Buffered %3.0f ms \xc2\xb7 a/v %+.1f ms \xc2\xb7 drops %" PRId64
                  " \xc2\xb7 underruns %" PRIu64,
                  view_audio_buffered(&st->audio) * 1000.0,
                  st->sync.diff * 1000.0, st->sync.dropped, st->audio.underruns);
    }

    view_panel_clip(st);

    /* Every section from here down is a header - a bar and its switch -
     * and a body under it.
     *
     * The white balance is offered for both kinds of clip, but only a
     * developed clip's can be switched off: a raw clip's is always applied,
     * so its header carries no switch. A developed clip has no mosaic, so
     * Debayer, Denoise, Lens, Rotate and Sharpen are not offered for one
     * at all. */
    if (view_section_header(st, VIEW_SEC_WB, "White balance",
                            st->dec.raw ? NULL : &st->sections.wb, false)) {
        view_panel_wb(st);
        view_section_end();
    }

    if (view_section_header(st, VIEW_SEC_EXPOSURE, "Exposure",
                            &st->sections.exposure, false)) {
        view_panel_exposure(st);
        view_section_end();
    }

    if (st->dec.raw &&
        view_section_header(st, VIEW_SEC_DEBAYER, "Debayer", NULL, false)) {
        static char names[8][16];
        static const char *labels[8];
        int n = CRAW_MIN((int)CRAW_DEBAYER_NB, 8), sel = lk->debayer;

        for (int i = 0; i < n; i++) {
            view_title_case(view_debayer_names[i], names[i], sizeof(names[i]));
            labels[i] = names[i];
        }
        view_prop("Method");
        if (view_segmented("debayer", labels, n, &sel))
            lk->debayer = (enum CRAWDebayer)sel;
        /* False colour suppression, DESIGN.md 17: median passes over the
         * colour differences after the debayer, which is why it is this
         * section's row and not a section of its own. Zero is off and
         * costs nothing; the pipeline always carries the stage. */
        view_slider_int("False colour", &lk->false_color, 0,
                        CRAW_FALSE_COLOR_MAX_PASSES, lk->false_color ? "%d passes" : "off");
        /* Defringe, DESIGN.md 19, after the debayer like the false colour
         * passes. */
        view_slider("Defringe", &lk->defringe, 0.0f, 1.0f, "%.2f");
        /* Highlight reconstruction, DESIGN.md 18: how the mosaic's clipped
         * samples come off, before the debayer, so a row here too. */
        view_check_row("Highlights", &lk->highlights, "Clipped ones rebuilt",
                       "Clipped ones cut");
        view_section_end();
    }

    if (st->dec.raw &&
        view_section_header(st, VIEW_SEC_DENOISE, "Denoise", &st->denoise.on, false)) {
        view_panel_denoise(st);
        view_section_end();
    }

    /* The lens switch is the one that can be refused: a clip whose frames
     * carry no correction record cannot be corrected, so the switch greys
     * out, showing off, rather than lying about what it would do. */
    if (st->dec.raw &&
        view_section_header(st, VIEW_SEC_LENS, "Lens", &st->lens.on,
                            !st->lens.record.valid)) {
        view_panel_lens(st);
        view_section_end();
    }

    if (st->dec.raw &&
        view_section_header(st, VIEW_SEC_ROTATE, "Rotate", &st->rotate.on, false)) {
        view_panel_rotate(st);
        view_section_end();
    }

    if (st->dec.raw &&
        view_section_header(st, VIEW_SEC_SHARPEN, "Sharpen", &st->sharpen.on, false)) {
        view_panel_sharpen(st);
        view_section_end();
    }

    if (view_section_header(st, VIEW_SEC_LOOK, "Look", &st->sections.look,
                            view_export_is_scene(st))) {
        /* Tone mapping, peak detection and gamut mapping used to be here.
         * They are not a look: the window is mapped with libplacebo's own
         * defaults for whatever the compositor negotiated, and choosing is
         * the export's business. The Export section has them. */
        char peak[64];
        bool grey;

        /* The curve the window is tone mapped with, out of libplacebo's
         * own list. A display setting rather than a look: it is above the
         * greying and outside the section's switch, because the window is
         * mapped through it whatever else is bypassed, and every
         * display-referred export follows it (gpu.c). */
        {
            static const char *names[32];
            static int nb_names;
            int tm;

            if (!nb_names) {
                nb_names = CRAW_MIN(view_tone_map_count(), 32);
                for (int i = 0; i < nb_names; i++)
                    names[i] = view_tone_map_name(i);
            }

            tm = CRAW_CLAMP(lk->tone_map, 0, nb_names - 1);
            if (view_combo_row("Tone map", &tm, names, nb_names, 16))
                lk->tone_map = tm;
        }

        /* A readout and not a control: what libplacebo measured for the
         * window against what the pipeline can have produced. */
        if (st->gpu.measured_peak > 0.0)
            snprintf(peak, sizeof(peak), "%.0f of %.0f nits",
                     st->gpu.measured_peak, view_peak_ceiling(st));
        else
            snprintf(peak, sizeof(peak), "Ceiling %.0f nits", view_peak_ceiling(st));
        view_prop_value("Display peak", peak);

        grey = view_grey_begin(st,
            "Not applied: the export is scene referred, so the film "
            "simulation is off and the screen shows what the file holds.");

        view_panel_film_sim(st);

        /* A user .cube comes from --lut on the command line; the panel only
         * shows it, and its two settings, while one is loaded. */
        if (st->lut && lk->film_sim < 0) {
            const char *base = strrchr(lk->lut_path, '/');
            static const char *lut_types[] = { "Native", "Normalized" };
            int sel = lk->lut_type == PL_LUT_NORMALIZED ? 1 : 0;

            view_prop_value("Cube", base ? base + 1 : lk->lut_path);

            view_prop("LUT type");
            if (view_segmented("luttype", lut_types, 2, &sel))
                lk->lut_type = sel ? PL_LUT_NORMALIZED : PL_LUT_NATIVE;

            sel = lk->lut_space;
            if (view_combo_row("LUT input", &sel, view_lut_space_names,
                               view_nb_lut_spaces) &&
                sel != lk->lut_space) {
                lk->lut_space = sel;
                view_lut_apply_space(st->lut, sel);
            }
        }

        if (st->lut_error[0])
            view_note_wrapped(st->lut_error, view_col_warn);

        view_grey_end(grey);
        view_section_end();
    }

    if (view_section_header(st, VIEW_SEC_ADJUST, "Adjustments",
                            &st->sections.adjust, view_export_is_scene(st))) {
        view_panel_adjust(st);
        view_section_end();
    }

    if (view_section_header(st, VIEW_SEC_GRAIN, "Film grain",
                            &st->grain.on, view_export_is_scene(st))) {
        view_panel_grain(st);
        view_section_end();
    }

    /* The switch is lit while the picture is cropped or a crop is being
     * drawn, so a rectangle in use never sits under a switch that says off.
     * A click turns off what is lit, the drawing first: in the middle of
     * one it ends it and keeps what was drawn, which is what `c` does, and
     * with a crop in use and nothing being drawn it drops the crop and the
     * whole frame is back. From off it starts drawing, as `c` does, with
     * the whole frame to draw on. Settled before the body is laid out so
     * that the readout under the header never shows the old rectangle for
     * a frame. */
    {
        int cx, cy, cw, ch;
        bool was = st->crop.on || view_crop_rect(st, &cx, &cy, &cw, &ch);
        bool lit = was;
        bool open = view_section_header(st, VIEW_SEC_CROP, "Crop", &lit, false);

        if (lit != was) {
            if (!was) {
                st->crop.on = true;
                view_crop_clear(st);
            } else if (st->crop.on) {
                st->crop.on = false;
            } else {
                view_crop_clear(st);
            }
        }
        if (open) {
            view_panel_crop(st);
            view_section_end();
        }
    }

    /* Two exports, two sections, two files: the image of the frame on
     * screen, and the video of the range. Neither field names the other's
     * file, and each keeps what was typed into it while its section is
     * folded: the fields are state, and a body that is not laid out simply
     * is not looked at. */
    if (view_section_header(st, VIEW_SEC_EXPORT_IMAGE, "Export image", NULL, false)) {
        view_panel_export_still(st);
        view_section_end();
    }

    /* A running video export holds its section open: the panel is locked
     * from the top down while the run is on, so the bar takes no click,
     * and the stop button and the progress are in this body. */
    if (st->video.active)
        st->sections.open[VIEW_SEC_EXPORT_VIDEO] = true;
    if (view_section_header(st, VIEW_SEC_EXPORT_VIDEO, "Export video", NULL, false)) {
        view_panel_export_video(st);
        view_section_end();
    }

    view_panel_bin(st);

    ImGui::EndDisabled();
}

/* ------------------------------------------------------- the crop shade */

/* Everything outside the crop, darkened, with a thin light border on the
 * rectangle itself. It is drawn under the panel on ImGui's background
 * list rather than anywhere in the render path, so the picture is not
 * touched and no shader changes; the geometry comes from the same view
 * transform that gives libplacebo its crop rects, which is what keeps the
 * shade on the same source pixels at every zoom, pan and window size.
 *
 * The bands are cut from whole pixel edges worked out once: two bands that
 * share an edge have to be handed the very same number, or the pixel on it
 * is shaded twice or not at all, which is a bright one pixel line all the
 * way across the picture. Everything is clipped to the picture as it lands
 * on screen, so a crop scrolled off an edge collapses the bands on that
 * side and its border stays off the picture rather than coming to rest on
 * the edge of the view. */
static void view_crop_draw(struct view_state *st)
{
    ImDrawList *out = ImGui::GetBackgroundDrawList();
    /* Black at 36 % coverage, which leaves 0.64 of the encoded picture
     * outside the rectangle - 82 where the picture is 128, and 41 where it
     * is 64: dark enough to read as excluded without hiding what is in it.
     * The overlay is composited premultiplied, in the swapchain's own
     * encoding. */
    const ImU32 shade = IM_COL32(0, 0, 0, 92);
    pl_rect2df src, dst, crop;
    int dx0, dy0, dx1, dy1, cx0, cy0, cx1, cy1, rx0, ry0, rx1, ry1;

    /* A clip's own rectangle is the picture on screen (view_disp_rect()),
     * so there is nothing outside it to shade and its border would be a
     * light line round the edge of the picture. */
    if (!st->crop.have || st->crop.from_clip || !view_crop_screen(st, &crop))
        return;

    view_zoom_rects(st, &src, &dst);
    if (pl_rect_w(dst) < 4.0f || pl_rect_h(dst) < 4.0f)
        return;

    dx0 = (int)lroundf(dst.x0);
    dy0 = (int)lroundf(dst.y0);
    dx1 = (int)lroundf(dst.x1);
    dy1 = (int)lroundf(dst.y1);
    rx0 = (int)lroundf(crop.x0);
    ry0 = (int)lroundf(crop.y0);
    rx1 = (int)lroundf(crop.x1);
    ry1 = (int)lroundf(crop.y1);
    cx0 = CRAW_CLAMP(rx0, dx0, dx1);
    cy0 = CRAW_CLAMP(ry0, dy0, dy1);
    cx1 = CRAW_CLAMP(rx1, cx0, dx1);
    cy1 = CRAW_CLAMP(ry1, cy0, dy1);

    out->PushClipRect(ImVec2((float)dx0, (float)dy0), ImVec2((float)dx1, (float)dy1));

    /* Four bands that tile the picture minus the rectangle. Any of them
     * can come out empty, and an empty one draws nothing. */
    out->AddRectFilled(ImVec2((float)dx0, (float)dy0), ImVec2((float)dx1, (float)cy0), shade);
    out->AddRectFilled(ImVec2((float)dx0, (float)cy1), ImVec2((float)dx1, (float)dy1), shade);
    out->AddRectFilled(ImVec2((float)dx0, (float)cy0), ImVec2((float)cx0, (float)cy1), shade);
    out->AddRectFilled(ImVec2((float)cx1, (float)cy0), ImVec2((float)dx1, (float)cy1), shade);

    /* The border on the crop's own edges rather than the clamped ones. */
    out->AddRect(ImVec2((float)rx0, (float)ry0), ImVec2((float)rx1, (float)ry1),
                 IM_COL32(235, 235, 240, 220), 0.0f, 0, 1.0f);

    out->PopClipRect();
}

/* A red frame round the picture while the file on screen is marked for
 * deletion (struct view_bin), so that a marked file cannot be looked at
 * without it being plain. Drawn inside the picture's own edges, on the
 * same whole pixels the crop shade uses: outside them it would fall in the
 * letterbox of a fitted picture but off the screen of a zoomed one, and
 * the frame has to be there whichever. */
static void view_bin_draw(struct view_state *st)
{
    ImDrawList *out = ImGui::GetBackgroundDrawList();
    const ImU32 red = IM_COL32(225, 40, 30, 255);
    const float t = 4.0f;
    pl_rect2df src, dst;
    float x0, y0, x1, y1;

    if (!st->bin.entries || !st->bin.entries[st->clip].marked || !st->have_frame)
        return;

    view_zoom_rects(st, &src, &dst);
    x0 = lroundf(dst.x0);
    y0 = lroundf(dst.y0);
    x1 = lroundf(dst.x1);
    y1 = lroundf(dst.y1);
    if (x1 - x0 < 4 * t || y1 - y0 < 4 * t)
        return;

    /* Four bands rather than a stroke, for the same reason the crop shade
     * is bands: whole pixels, and edges that meet exactly. */
    out->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y0 + t), red);
    out->AddRectFilled(ImVec2(x0, y1 - t), ImVec2(x1, y1), red);
    out->AddRectFilled(ImVec2(x0, y0 + t), ImVec2(x0 + t, y1 - t), red);
    out->AddRectFilled(ImVec2(x1 - t, y0 + t), ImVec2(x1, y1 - t), red);
}

void view_ui_build(struct view_state *st)
{
    ImGuiIO &io = ImGui::GetIO();
    int64_t now = av_gettime_relative();

    io.DisplaySize = ImVec2((float)st->gpu.ui_w, (float)st->gpu.ui_h);
    io.DeltaTime = view_ui_last ? CRAW_MAX((float)(now - view_ui_last) / 1.0e6f, 1.0e-4f)
                                : 1.0f / 60.0f;
    view_ui_last = now;

    st->menu.anchored = false;

    ImGui::NewFrame();

    /* Under the panel, on the background list. */
    view_crop_draw(st);
    view_bin_draw(st);

    /* The panel is a sidebar the window's full height against its right
     * edge, set off from the picture by a hairline. It keeps its place
     * under everything that opens over it - the menu's columns, a combo's
     * list - however it is clicked. */
    ImGui::SetNextWindowPos(ImVec2((float)st->gpu.ui_w - VIEW_UI_WIDTH, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(VIEW_UI_WIDTH, (float)st->gpu.ui_h));
    if (ImGui::Begin("##cineview", NULL,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav)) {
        ImDrawList *dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetWindowPos();

        dl->PushClipRectFullScreen();
        dl->AddRectFilled(p, ImVec2(p.x + 1.0f, p.y + ImGui::GetWindowHeight()),
                          ImGui::GetColorU32(view_col_border));
        dl->PopClipRect();

        view_panel(st);
    }
    ImGui::End();

    /* The menu's columns are built after the panel, which is what puts
     * them on top of it. The row scrolling out of view, or its whole
     * section collapsing, takes the menu with it rather than leaving it
     * hanging off nothing. */
    if (st->menu.open && !st->menu.anchored)
        view_ui_menu_close(st);
    view_menu_frame(st);

    ImGui::Render();

    /* Whether a field had the keyboard this frame - a file name, or a
     * number being typed into - is known once the frame is over, and SDL3
     * sends no text events to a window that has not started text input,
     * so the answer goes straight back to SDL here. A number field has to
     * take the keys the way a text field does, or a typed frame number
     * would go to the shortcuts - `1` is the 1:1 zoom. */
    st->ui_edit_active = ImGui::GetCurrentContext()->WantTextInputNextFrame == 1;
    if (st->ui_edit_active != view_ui_text_on && view_ui_window) {
        view_ui_text_on = st->ui_edit_active;
        if (view_ui_text_on)
            SDL_StartTextInput(view_ui_window);
        else
            SDL_StopTextInput(view_ui_window);
    }
}
