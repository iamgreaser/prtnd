# vim: set sts=4 sw=4 et :

"""
PRTND: Please Run Tomorrow Never Dies
A posthumous load remover

Copyright (c) 2020 Ben "GreaseMonkey" Russell

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not
       be misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
       distribution.
"""

import sys

import cv2 as cv
import numpy as np


class Application:
    __slots__ = (
        "_vid_fname",
        "_vid_cap",
        "_im_layout_mask",
        "_im_loading_text",
        "_last_im",
        "_loading_match_boxes",
        "_im_snap",
        "_snap_mouse_pt0",
        "_snap_mouse_pt1",
        "_last_raw_frame",
        "_last_frame_pos",
        "_is_playing",
        "_was_loading",
        "_load_measure_beg",
        "_load_measure_end",
        "_load_accum_frames",
    )

    def __init__(self, vid_fname: str) -> None:
        self._vid_fname = vid_fname
        self._vid_cap = None
        self._im_layout_mask = None
        self._im_loading_text = None
        self._loading_match_boxes = []
        self._last_im = None
        self._im_snap = None
        self._snap_mouse_pt0 = None
        self._snap_mouse_pt1 = None
        self._last_raw_frame = None
        self._last_frame_pos = -2
        self._is_playing = False
        self._was_loading = False
        self._load_measure_beg = None
        self._load_measure_end = None
        self._load_accum_frames = 0

    def run(self) -> None:
        self._vid_cap = cv.VideoCapture(self._vid_fname)

        self.init_windows()

        while True:
            k = cv.waitKey(1)
            if k in [ord("q"), ord("Q"), 27]:
                return

            self.do_tick(k)

    def do_tick(self, k: int) -> None:
        if k in [ord("p"), ord("P")]:
            if self._is_playing and self._load_accum_frames != 0:
                dt = self._load_accum_frames
                self._load_accum_frames = 0
                raw_fps = int(round(self._vid_cap.get(cv.CAP_PROP_FPS)))
                fps = float(raw_fps)
                dt_sec = float(dt/fps)
                print("Accumulated loading time: %7d frames / %9.3f seconds @ %7.3f FPS" % (
                    dt, dt_sec, fps,
                ))

            self._is_playing = not self._is_playing

        if self._is_playing:
            frame_pos = int(round(self._vid_cap.get(cv.CAP_PROP_POS_FRAMES)))
            cv.setTrackbarPos("frame", "vid", frame_pos)
            got, im = self._vid_cap.read()
            if got:
                self._last_raw_frame = im
                self._last_frame_pos = frame_pos

        else:
            frame_pos = cv.getTrackbarPos("frame", "vid")
            if self._last_frame_pos != frame_pos:
                self._vid_cap.set(cv.CAP_PROP_POS_FRAMES, frame_pos)
                got, im = self._vid_cap.read()
                if got:
                    self._last_raw_frame = im
                    self._last_frame_pos = frame_pos
            else:
                im = self._last_raw_frame
                got = True

        if got:
            #im = im[:,380:] # Liam's layout 1280x720
            #im = im[:,288:] # matimbre's layout 1280x720
            #im = im[:,330:] # OllieNK's layout 854x480

            #im = cv.resize(im, (640,480)) # 4:3
            im = cv.resize(im, (640,360)) # 16:9
            self.do_image(frame_pos, im, k)

    def init_windows(self) -> None:
        cv.namedWindow("vid")
        frame_count = int(round(self._vid_cap.get(cv.CAP_PROP_FRAME_COUNT)))
        cv.createTrackbar("frame", "vid", 0, frame_count-1, lambda x=None:None)

    def do_image(self, frame_pos: int, im: np.array, k: int) -> None:
        need_layout_mask = (k in [ord("l"), ord("L")])
        need_snapshot = (k in [ord("s"), ord("S")])

        im_yuv = cv.cvtColor(im, cv.COLOR_BGR2YUV)

        if need_snapshot:
            self._im_snap = im.copy()
            cv.namedWindow("snapshot")
            cv.setMouseCallback("snapshot", self.on_snap_mouse)

        if self._im_snap is not None:
            im_snap_output = self._im_snap.copy()

            if self._snap_mouse_pt0 is not None and self._snap_mouse_pt1 is not None:
                cv.rectangle(im_snap_output, self._snap_mouse_pt0, self._snap_mouse_pt1, (0,255,0),2)

            cv.imshow("snapshot", im_snap_output)

        load_match = False
        for match_box in self._loading_match_boxes:
            load_match = (load_match or match_box.did_match(
                src_im_yuv=im_yuv,
            ))

        if (not self._is_playing) or self._last_im is None or not load_match:
            self._last_im = im.copy()
            if self._was_loading:
                self._was_loading = False
                try:
                    dt = int(self._load_measure_end - self._load_measure_beg + 1)
                    raw_fps = int(round(self._vid_cap.get(cv.CAP_PROP_FPS)))
                    fps = float(raw_fps)
                    dt_sec = float(dt/fps)
                    self._load_accum_frames += dt
                    print("Loading time: %7d frames / %9.3f seconds @ %7.3f FPS" % (
                        dt, dt_sec, fps,
                    ))
                finally:
                    self._load_measure_beg = None
                    self._load_measure_end = None
        else:
            if not self._was_loading:
                self._was_loading = True
                self._load_measure_beg = frame_pos
                self._load_measure_end = frame_pos
            else:
                self._load_measure_beg = min(self._load_measure_beg, frame_pos)
                self._load_measure_end = max(self._load_measure_end, frame_pos)

        im_output = self._last_im.copy()

        w, h, = im_output.shape[:2][::-1]

        if load_match:
            # Loading
            cv.circle(im_output, (15,15), 10, (0,0,255), -1)
        else:
            # Running
            cv.circle(im_output, (15,15), 10, (0,255,0), -1)

        cv.imshow("vid", im_output)

    def on_snap_mouse(self, event, x, y, flags, param) -> None:
        if event == cv.EVENT_LBUTTONDOWN:
            self._snap_mouse_pt0 = (x, y,)
            self._snap_mouse_pt1 = (x, y,)

        elif event == cv.EVENT_MOUSEMOVE:
            if self._snap_mouse_pt0 is not None:
                self._snap_mouse_pt1 = (x, y,)

        elif event == cv.EVENT_LBUTTONUP:
            try:
                self._snap_mouse_pt1 = (x, y,)
                x0, y0, = self._snap_mouse_pt0
                x1, y1, = self._snap_mouse_pt1
                self._loading_match_boxes.append(
                    MatchBox(
                        src_im_yuv=self._im_snap,
                        pt0=self._snap_mouse_pt0,
                        pt1=self._snap_mouse_pt1,
                    ),
                )
            finally:
                self._snap_mouse_pt0 = None
                self._snap_mouse_pt1 = None


class MatchBox:
    __slots__ = (
        "_im_yuv",
        "_pt0",
        "_pt1",
    )

    def __init__(self, *, src_im_yuv, pt0, pt1) -> None:
        x0, y0, = pt0
        x1, y1, = pt1
        im_yuv = src_im_yuv[y0:y1,x0:x1]
        im_yuv = cv.cvtColor(im_yuv, cv.COLOR_BGR2YUV)
        self._pt0 = (x0-10, y0-10)
        self._pt1 = (x1+10, y1+10)
        self._im_yuv = im_yuv

    def did_match(self, src_im_yuv) -> bool:
        x0, y0, = self._pt0
        x1, y1, = self._pt1
        im_scan_sub = src_im_yuv[y0:y1,x0:x1]
        res = cv.matchTemplate(im_scan_sub, self._im_yuv, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        return (max_val >= 0.6)


if __name__ == "__main__":
    Application(*sys.argv[1:]).run()
