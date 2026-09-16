/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* --trace FILE: what the run loop actually did, iteration by iteration.
 *
 * Playback jitter is a question about when things happened, not about how
 * long they took on average, so the smoothed averages -v already prints
 * cannot answer it. One record per iteration that reached the swapchain is
 * kept in memory - a write inside the loop would be measuring the writing -
 * and the whole run goes out at exit, followed by the summary the record
 * stream supports: the present interval and its spread, how many refreshes
 * each picture was left on screen, how late each one was against the master
 * clock, and how many display passes drew a picture that had not changed. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <libavutil/time.h>

#include <libplacebo/dispatch.h>

#include "view.h"

/* Twenty minutes of a 60 Hz loop, at about a hundred bytes a record.
 * Recording stops there rather than wrapping, so that no summary is ever
 * computed over a window that moved while it was being filled. */
#define VIEW_TRACE_MAX (72 * 1024)

static char view_trace_path[VIEW_MAX_PATH];

/* Where the GPU time of a display pass goes, one row per shader libplacebo
 * dispatched. The signature is libplacebo's own: one number per distinct
 * compiled shader, which is what makes the scaler, the peak detector, each
 * hook and the overlay blend separate rows without this file knowing
 * anything about them. */
#define VIEW_TRACE_PASSES 64

static struct {
    uint64_t sig;
    char name[120];
    uint64_t ns;
    int64_t n;
} view_trace_pass_tab[VIEW_TRACE_PASSES];
static int view_trace_nb_passes;

/* libplacebo calls this for every pass of every pl_render_image on the
 * window's renderer. `last` is that pass's own GPU time, from the timer
 * queries libplacebo keeps per pass, and arrives a few frames after the
 * dispatch that produced it - which is why this is a total and not a
 * per-frame column. */
void view_trace_pass(void *priv, const struct pl_render_info *info)
{
    const struct pl_dispatch_info *d = info->pass;
    int i;

    (void)priv;

    if (!d || !d->num_samples)
        return;

    for (i = 0; i < view_trace_nb_passes; i++)
        if (view_trace_pass_tab[i].sig == d->signature)
            break;
    if (i == view_trace_nb_passes) {
        if (i == VIEW_TRACE_PASSES)
            return;
        view_trace_nb_passes++;
        view_trace_pass_tab[i].sig = d->signature;
        snprintf(view_trace_pass_tab[i].name, sizeof(view_trace_pass_tab[i].name),
                 "%s", d->shader && d->shader->description
                       ? d->shader->description : "(unnamed)");
    }

    for (int k = 0; k < d->num_samples; k++) {
        view_trace_pass_tab[i].ns += d->samples[k];
        view_trace_pass_tab[i].n++;
    }
}

static void view_trace_passes(FILE *o, double dur)
{
    uint64_t total = 0;

    if (!view_trace_nb_passes)
        return;

    for (int i = 0; i < view_trace_nb_passes; i++)
        total += view_trace_pass_tab[i].ns;

    fprintf(o, "trace: the window's render, by libplacebo pass "
               "(%.1f ms of GPU in %.2f s):\n", (double)total / 1.0e6, dur);

    /* A selection sort over at most sixty-four rows, printed straight out:
     * sorting them into a copy would only be another array to get wrong. */
    bool done[VIEW_TRACE_PASSES] = { false };

    for (int printed = 0; printed < view_trace_nb_passes; printed++) {
        int best = -1;

        for (int i = 0; i < view_trace_nb_passes; i++) {
            if (done[i] || !view_trace_pass_tab[i].n)
                continue;
            if (best < 0 || view_trace_pass_tab[i].ns > view_trace_pass_tab[best].ns)
                best = i;
        }
        if (best < 0)
            break;

        fprintf(o, "trace:   %7.3f ms/dispatch  %7.1f ms total  %5" PRId64
                   "x  %s\n",
                (double)view_trace_pass_tab[best].ns /
                (double)view_trace_pass_tab[best].n / 1.0e6,
                (double)view_trace_pass_tab[best].ns / 1.0e6,
                view_trace_pass_tab[best].n, view_trace_pass_tab[best].name);
        done[best] = true;
    }
}

int view_trace_open(struct view_state *st, const char *path, double refresh_hz)
{
    struct view_trace *tr = &st->trace;

    tr->recs = calloc(VIEW_TRACE_MAX, sizeof(*tr->recs));
    if (!tr->recs)
        return CRAW_ERROR(ENOMEM);

    snprintf(view_trace_path, sizeof(view_trace_path), "%s", path);
    tr->refresh_hz = refresh_hz;
    tr->max = VIEW_TRACE_MAX;
    tr->nb = 0;
    tr->t0 = av_gettime_relative();
    tr->on = true;

    return 0;
}

void view_trace_iter(struct view_state *st, int64_t now)
{
    struct view_trace *tr = &st->trace;

    if (!tr->on)
        return;

    /* A record covers everything from the iteration that began a picture's
     * work to the present that put it on screen, which since the picture
     * pass moved off the present path is several iterations: the decode,
     * the pipeline submit and the render all happen while the picture
     * before it is still being held. So the record is only started when the
     * last one has been written out. */
    if (!tr->cur.t_iter) {
        tr->cur.t_iter = now;
        tr->cur.index = -1;
    }
    st->gpu.rec = &tr->cur;
}

void view_trace_push(struct view_state *st)
{
    struct view_trace *tr = &st->trace;

    if (!tr->on)
        return;

    st->gpu.rec = NULL;

    if (tr->nb >= tr->max) {
        tr->cur = (struct view_trace_rec) { 0 };
        return;
    }

    /* libcraw's own timestamps come back VIEW_SLOTS submissions later, so
     * the measurement that has just landed belongs to a record already
     * written. Walking back to it is what makes the craw column the GPU time
     * of the picture on that line rather than of some picture near it. */
    if (st->gpu.last_gpu_ms > 0.0) {
        int seen = 0;

        for (int i = tr->nb - 1; i >= 0; i--) {
            if (!tr->recs[i].stepped)
                continue;
            if (++seen == VIEW_SLOTS) {
                tr->recs[i].craw_gpu_ms = st->gpu.last_gpu_ms;
                break;
            }
        }
    }

    tr->recs[tr->nb++] = tr->cur;
    tr->cur = (struct view_trace_rec) { 0 };
}

static double view_trace_at(const struct view_trace *tr, int64_t t)
{
    return t ? (double)(t - tr->t0) / 1000.0 : 0.0;
}

static double view_trace_ms(int64_t a, int64_t b)
{
    return a && b ? (double)(b - a) / 1000.0 : 0.0;
}

static void view_trace_col(FILE *f, const struct view_trace *tr, int64_t t)
{
    if (t)
        fprintf(f, "%9.3f ", view_trace_at(tr, t));
    else
        fprintf(f, "%9s ", "-");
}

static void view_trace_lines(struct view_state *st, FILE *f)
{
    const struct view_trace *tr = &st->trace;

    fprintf(f, "# cineview playback trace\n"
               "# times are milliseconds from the start of the run; the "
               "durations say ms\n"
               "#\n"
               "# t_iter   the run loop's iteration began\n"
               "# pkt      this picture's packet reached the reader's queue\n"
               "# step     view_step_frame() returned: decode done, libcraw "
               "submitted\n"
               "# submit   libcraw's vkQueueSubmit2() returned\n"
               "# fence    that submission's fence was first seen signalled, "
               "polled once an iteration\n"
               "# present  pl_swapchain_swap_buffers() returned\n"
               "# dt       present to present\n"
               "# wait     blocked in pl_swapchain_start_frame, i.e. no "
               "swapchain image was free\n"
               "# img      pl_render_image itself\n"
               "# hdr      pl_renderer_get_hdr_metadata\n"
               "# rend     the swapchain submit\n"
               "# pres     the swap itself, i.e. the wait for a free buffer\n"
               "# craw     libcraw's own GPU timestamps for that picture\n"
               "# fwait    blocked on a libcraw fence this iteration\n"
               "# a/v      master clock minus the picture's pts at the "
               "present, positive is late\n"
               "# new      a new picture was decoded and developed\n"
               "# drew     the picture pass ran, rather than a cached "
               "picture being composed\n\n");

    fprintf(f, "%8s %7s %9s %9s %9s %9s %9s %9s %7s %6s %6s %6s %6s %6s %6s %8s %3s %4s\n",
            "index", "pts", "t_iter", "pkt", "step", "submit", "fence",
            "present", "dt", "wait", "img", "hdr", "rend", "craw", "fwait",
            "a/v", "new", "drew");

    for (int i = 0; i < tr->nb; i++) {
        const struct view_trace_rec *r = &tr->recs[i];

        fprintf(f, "%8" PRId64 " %7.3f ", r->index, r->pts);
        view_trace_col(f, tr, r->t_iter);
        view_trace_col(f, tr, r->t_pkt);
        view_trace_col(f, tr, r->t_step);
        view_trace_col(f, tr, r->t_submit);
        view_trace_col(f, tr, r->t_fence);
        view_trace_col(f, tr, r->t_present);
        fprintf(f, "%7.3f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %8.2f %3d %4d\n",
                i ? view_trace_ms(tr->recs[i - 1].t_present, r->t_present) : 0.0,
                view_trace_ms(r->t_disp, r->t_start),
                view_trace_ms(r->t_start, r->t_image),
                view_trace_ms(r->t_image, r->t_hdr),
                view_trace_ms(r->t_hdr, r->t_render),
                r->craw_gpu_ms, r->fence_ms,
                (r->clock - r->pts) * 1000.0, r->stepped, r->rendered);
    }
}

void view_trace_close(struct view_state *st)
{
    struct view_trace *tr = &st->trace;
    FILE *f;
    double dur, sum = 0.0, sumsq = 0.0, mean = 0.0, sd = 0.0, late_worst = 0.0;
    double frame_dur = st->dec.frame_rate.num ? 1.0 / av_q2d(st->dec.frame_rate)
                                              : 1.0 / 29.97;
    /* How long each picture was left on screen, in refreshes of the
     * output: the interval between the present that put it up and the one
     * that replaced it. This is what the eye sees. It is not the same as
     * how many times a picture was presented - a picture presented once
     * stays on screen until the next present, however long that is - and
     * counting the presents was what made an early version of this summary
     * flatter itself. */
    double refresh_ms = tr->refresh_hz > 1.0 ? 1000.0 / tr->refresh_hz : 0.0;
    int64_t onscreen[10] = { 0 }, shown = 0;
    int64_t first_present = 0;
    double gap_sum = 0.0, gap_sumsq = 0.0;
    int64_t intervals = 0, frames = 0, late = 0, unchanged = 0, rendered = 0;
    int refresh[8] = { 0 };
    int run = 0;

    if (!tr->on)
        return;

    f = fopen(view_trace_path, "w");
    if (f)
        view_trace_lines(st, f);
    else
        fprintf(stderr, "--trace %s: cannot write\n", view_trace_path);

    for (int i = 0; i < tr->nb; i++) {
        const struct view_trace_rec *r = &tr->recs[i];

        if (i) {
            double dt = view_trace_ms(tr->recs[i - 1].t_present, r->t_present);

            sum += dt;
            sumsq += dt * dt;
            intervals++;
        }

        if (r->rendered) {
            rendered++;
            if (!r->stepped)
                unchanged++;
        }

        if (!r->stepped) {
            run++;
            continue;
        }

        /* The picture that was on screen until this one arrived was shown
         * for `run` presents. The first one is skipped: nothing was on
         * screen before it. */
        if (frames > 0)
            refresh[run < 7 ? run : 7]++;
        run = 1;
        frames++;

        if (first_present) {
            double gap = view_trace_ms(first_present, r->t_present);

            gap_sum += gap;
            gap_sumsq += gap * gap;
            shown++;
            if (refresh_ms > 0.0) {
                int n = (int)(gap / refresh_ms + 0.5);

                onscreen[CRAW_CLAMP(n, 1, 9)]++;
            }
        }
        first_present = r->t_present;

        if (r->clock - r->pts > 0.5 * frame_dur)
            late++;
        late_worst = CRAW_MAX(late_worst, r->clock - r->pts);
    }
    if (frames > 0 && run > 0)
        refresh[run < 7 ? run : 7]++;

    if (intervals) {
        mean = sum / (double)intervals;
        sd = sqrt(CRAW_MAX(sumsq / (double)intervals - mean * mean, 0.0));
    }
    dur = tr->nb > 1 ? view_trace_ms(tr->recs[0].t_iter,
                                     tr->recs[tr->nb - 1].t_present) / 1000.0
                     : 0.0;

    for (int pass = 0; pass < 2; pass++) {
        FILE *o = pass ? stderr : f;

        if (!o)
            continue;

        fprintf(o, "%strace: %.2f s, %d presents (%.2f/s), %" PRId64
                   " pictures (%.2f/s)\n", pass ? "" : "\n", dur, tr->nb,
                dur > 0.0 ? (double)tr->nb / dur : 0.0,
                frames, dur > 0.0 ? (double)frames / dur : 0.0);
        fprintf(o, "trace: present interval mean %.3f ms, sd %.3f ms\n", mean, sd);
        fprintf(o, "trace: presents per picture:");
        for (int i = 1; i < 8; i++)
            if (refresh[i])
                fprintf(o, " %d%s:%d", i, i == 7 ? "+" : "", refresh[i]);
        fprintf(o, "\n");

        if (shown > 1) {
            double gm = gap_sum / (double)shown;
            double gsd = sqrt(CRAW_MAX(gap_sumsq / (double)shown - gm * gm, 0.0));

            fprintf(o, "trace: one picture to the next: mean %.3f ms, "
                       "sd %.3f ms\n", gm, gsd);
            if (refresh_ms > 0.0) {
                fprintf(o, "trace: refreshes on screen per picture (%.3f Hz "
                           "output):", tr->refresh_hz);
                for (int i = 1; i < 10; i++)
                    if (onscreen[i])
                        fprintf(o, " %d%s:%" PRId64 " (%.1f%%)", i,
                                i == 9 ? "+" : "", onscreen[i],
                                100.0 * (double)onscreen[i] / (double)shown);
                fprintf(o, "\n");
            }
        }
        fprintf(o, "trace: %" PRId64 " late (over half a frame behind the clock), "
                   "worst %+.1f ms; %" PRId64 " dropped\n",
                late, late_worst * 1000.0, st->sync.dropped);
        fprintf(o, "trace: %.2f iterations per picture; %" PRId64 " of %" PRId64
                   " picture passes drew a picture that had not changed\n",
                frames ? (double)tr->nb / (double)frames : 0.0,
                unchanged, rendered);
        if (tr->nb >= tr->max)
            fprintf(o, "trace: recording stopped at %d records\n", tr->max);
        view_trace_passes(o, dur);
    }

    if (f)
        fclose(f);

    free(tr->recs);
    tr->recs = NULL;
    tr->on = false;
}
