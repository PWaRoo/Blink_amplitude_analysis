"""
(*)~---------------------------------------------------------------------------
Blink analyses tool using eye-tracking data from pupil GUI by pupil-labs.com
- source data from pupil: original eye videos, data files eyeX_timestamps.npy,
blinks.csv, pupil_positions.csv, ...

this code
- extracts images from video files that are identified by blinks.csv as blinks
- uses extracted images to create a vertical slice containing pupil center
- finds empirical peak thresholds to determine palpebral aperture
- calculate relative blink amplitude
- saves determined values to blinks_detailed.csv

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
author: p.wagner@unsw.edu.au
---------------------------------------------------------------------------~(*)
"""

import numpy as np
import pandas as pd
import math as m
import cv2, os.path, glob, shutil
import matplotlib.pyplot as plt                 # required for individual display for data points
from scipy.signal import find_peaks

# TODO: bookmark 1 try binary image filter for lid_blink_ypos

# TODO: maybe apply curve fitting through found top lid positions during blinking to exclude outliers
#   can be done post processing using data from blinks_detailed.csv ["lid_blink_ypos"]

# TODO: bookmark 2 maybe extend blink detection until first few and last few images are of high confidence
#   replace with current procedure to just add a few frames prior and past
#   downside: lid positions pre and post maybe unreliable since gaze position could have change a lot

# TODO: change flow to finish one by one recording
# TODO: current eye video source resolution 196x196 >> seek independence form resolution
# TODO: write blink values into blink images

# TODO: done    adjust blink folder naming to match blink id currently start with 0000 and blink id with 1
# TODO: done    blink sorting when changing decimals sorted(...key=lambda x: int(x.split('\\')[-1].split('_')[-1][0:-4])
# TODO: done    record amplitude data OD OS,
# TODO: done    run imaging of blinks
# TODO: done    employ filters and test for best outcome -> pass on blink csv file
# TODO: done    find blink amplitude of eye in blink summary images
# TODO: done    eye video source switch to original eye video, find out what is the difference?
# TODO: done    pupil centre for image slicing, 2 consecutive values with average confidence required > 0.8
# TODO: done    check if eye data for eye_ids is available
# TODO: done    optional: delete blink folders and single blink images
# TODO: done    working fine saving, lid_blink_ypos in blinks_details.csv not working correct - not all values get saved
# TODO: done    check if source files are available
# TODO: done    check which eye is available

# TODO: skip   check of double blinks and false detected blinks

# set up class Pupil Labs Helpers [PLH]
# extract_blink_images = True - extract blink related images for eye_ids and store in separate folders
# required source data: eye?_timestamps.npy and original eye videos
# see below for further options for data validation

#  - requires a 196 x 196 video frame

class PLH(object):  # class Pupil Labs Helpers

    def find_all_recordings(recordings_location):

        meta_data_columns = ['Data Format Version',
                             'Export Date',
                             'Relative Time Range',
                             'date',
                             'recording',
                             'export',
                             'history_length',
                             'onset_confidence_threshold',
                             'offset_confidence_threshold',
                             'blinks_exported',
                             'image_extraction_eye0',
                             'image_extraction_eye1',
                             'file_path']

        meta_data_all = pd.DataFrame(columns=meta_data_columns)

        # find all info.csv files in subfolders of recording folder
        fn_search = str(recordings_location + '/*/*/info.csv')
        print(fn_search)
        for file in glob.glob(fn_search):
            export_reports = glob.glob(str(file[:-9] + '/*/*/export_info.csv'))
            for export_report_fn in export_reports:
                meta_data = pd.DataFrame(columns=meta_data_columns)
                # file path of export
                fp = os.path.join(*export_report_fn.split('\\')[:-1])
                # export meta data
                export_meta_data = pd.read_csv(os.path.join(fp, 'export_info.csv'), index_col=0).T
                meta_data['Data Format Version'] = export_meta_data['Data Format Version']
                meta_data['Export Date'] = export_meta_data['Export Date']
                meta_data['Relative Time Range'] = export_meta_data['Relative Time Range']
                # date, recording, export
                meta_data['date'] = (fp.split('\\')[-4:-3])
                meta_data['recording'] = (fp.split('\\')[-3:-2])
                meta_data['export'] = (fp.split('\\')[-1:])
                meta_data['file_path'] = (fp)
                # blink meta data
                if not os.path.isfile(os.path.join(fp, 'blink_detection_report.csv')):
                    meta_data['history_length'] = "blink export missing"
                    meta_data['blinks_exported'] = 0
                else:
                    blink_meta_data = pd.read_csv(os.path.join(fp, 'blink_detection_report.csv'), index_col=0).T
                    meta_data['history_length'] = blink_meta_data['history_length']
                    meta_data['onset_confidence_threshold'] = blink_meta_data['onset_confidence_threshold']
                    meta_data['offset_confidence_threshold'] = blink_meta_data['offset_confidence_threshold']
                    print(fp)
                    print(blink_meta_data['blinks_exported'])
                    meta_data['blinks_exported'] = float(blink_meta_data['blinks_exported'])
                    if 'image_extraction_eye0' in blink_meta_data.columns:
                        meta_data['image_extraction_eye0'] = blink_meta_data['image_extraction_eye0']
                    if 'image_extraction_eye1' in blink_meta_data.columns:
                        meta_data['image_extraction_eye1'] = blink_meta_data['image_extraction_eye0']
                meta_data_all = meta_data_all.append((meta_data), ignore_index=True, sort=False)

        export_fn = recordings_location + "\\blink_meta_data_all.csv"
        meta_data_all[meta_data_columns].to_csv(export_fn, index=False)

        # select file path for image extraction
        fp_all = meta_data_all.loc[(meta_data_all.loc[:, 'blinks_exported'] > 0), 'file_path']
        return fp_all

    def check_source_f(file_path):
        # check if all source files are available
        print(file_path)
        fns = [
            os.path.join(file_path, 'blinks.csv'),
            os.path.join(file_path, 'pupil_positions.csv'),
        ]

        if not all([os.path.isfile(fn) for fn in fns]):
            for fn in fns:
                if not os.path.isfile(fn):
                    print(fn)
            exit("Source files missing ")

        # check which eye_ids are in pupil_positions.csv
        eye_id_data = pd.read_csv(os.path.join(file_path, "pupil_positions.csv"), nrows=10)
        eye_ids = (eye_id_data.loc[:, "eye_id"].unique())
        # check if eyeX.mp4 and eyeX_timestamps.npy is available
        raw_data_fns = []
        for eye_id in eye_ids:
            raw_data_fn = (os.path.abspath(os.path.join(file_path, ". ./..")) + "/eye" + str(eye_id) + ".mp4")
            raw_data_fns = np.append(raw_data_fns, raw_data_fn)
            raw_data_fn = (os.path.abspath(os.path.join(file_path, "../..")) + "/eye" + str(eye_id) + "_timestamps.npy")
            raw_data_fns = np.append(raw_data_fns, raw_data_fn)

        if not all([os.path.isfile(fn) for fn in raw_data_fns]):
            for fn in raw_data_fns:
                if not os.path.isfile(fn):
                    print(fn)
            exit("Source file(s) missing ")
        print("all source files available")
        return eye_ids

    def extract_blink_related_images(eye_ids, file_path, display_image_extraction, display_pupil_ellipse,
                                     display_pupil_details):
        raw_data_path = os.path.abspath(os.path.join(file_path,"../.."))
        print(raw_data_path)
        # Load blink data and flag in pupil_positions
        Bdata = pd.read_csv(os.path.join(file_path,  'blinks.csv'))
        # # select images in blink interval and save images
        for eye_id in eye_ids:
            # check if export has been done already and move on to the next if so
            b_meta_data = pd.read_csv(os.path.join(file_path, 'blink_detection_report.csv'), index_col=0).T
            if str('image_extraction_eye' + str(eye_id)) in b_meta_data:
                if b_meta_data[str('image_extraction_eye' + str(eye_id))].item() == 'completed':
                    print("extraction compleeted previously")
                    break
            # Load timestamps and pupil positions for eye0 or eye1, pupil positions are indexed by pupil_timestamp.
            if eye_id == 0:
                timestamps = np.load(raw_data_path + "/eye0_timestamps.npy")
                fn_eye_video = "eye0.mp4"
            elif eye_id == 1:
                timestamps = np.load(raw_data_path + "/eye1_timestamps.npy")
                fn_eye_video = "eye1.mp4"
            timestamps = np.round(timestamps, decimals=6)

            pupil_positions = pd.read_csv(os.path.join(file_path, "pupil_positions.csv"), index_col="pupil_timestamp")
            pupil_positions.index = np.round(pupil_positions.index, decimals=6)
            pupil_positions = pupil_positions[pupil_positions["eye_id"] == eye_id]

            # # Load blink data and flag in pupil_positions
            # Bdata = pd.read_csv(os.getcwd() + '/exports/000/blinks.csv')
            pupil_positions["blinks"] = np.nan
            pupil_positions["blinkNr"] = np.nan
            for row in Bdata.index:
                # flag pupil_positions during identified blinks with blink_idx
                # extend blinks by +/- 0.1 s

                blink_start_adjusted = \
                    pupil_positions.index[(pupil_positions.index >= (Bdata["start_timestamp"][row] - 0.05))][0]
                blink_end_adjusted = \
                    pupil_positions.index[(pupil_positions.index <= (Bdata["end_timestamp"][row]) + 0.05)][-1]

                pupil_positions.loc[(pupil_positions.index >= blink_start_adjusted) &
                                    (pupil_positions.index <= blink_end_adjusted), ["blinkNr"]] = Bdata.loc[row, 'id'] # +1 to match blink ids
            # Load video
            video = cv2.VideoCapture(os.path.join(raw_data_path, fn_eye_video))
            n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            while frame_idx < n_frames:
                # Read frame. Note: This will read frame-by-frame accurately!
                # Do not try to call video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)!
                success, frame = video.read()
                if not success:
                    print('video.read() error ' + str(frame_idx))
                else:
                    # Get corresponding pupil_time for frame_idx
                    pupil_time = timestamps[frame_idx]
                    # Get corresponding pupil positions for pupil_time
                    if pupil_time not in pupil_positions.index:
                        print("Unmatched timestamp:" + str(frame_idx) + " " + str(pupil_time))
                        pupil_data = None
                    else:
                        pupil_data = pd.DataFrame(pupil_positions.loc[pupil_time])
                        # Render ellipse from pupil positions data
                        if display_pupil_ellipse:
                            cv2.ellipse(
                                img=frame,
                                center=(
                                    int(pupil_data.loc["ellipse_center_x"]), int(pupil_data.loc["ellipse_center_y"])),
                                axes=(pupil_data.loc["ellipse_axis_a"] / 2, pupil_data.loc["ellipse_axis_b"] / 2),
                                angle=pupil_data.loc["ellipse_angle"],
                                startAngle=0,
                                endAngle=360,
                                color=(255, 0, 0),
                                thickness=1
                            )
                    # Flip and enlarge for better visibility
                    if eye_id == 0:
                        frame = cv2.flip(frame, -1)
                    frame = cv2.resize(frame, (384, 384))

                    # # write pupil_position infos onto corresponding video frame
                    if display_pupil_details:
                        if pupil_data is not None:
                            PLH.VideoFrameInfo(frame, pupil_data, frame_idx, pupil_time)
                        else:
                            print('no pupil data')
                    # identify flaged blinks and save all images in seperate folder
                    if pupil_data is not None:
                        # print(pupil_data.loc["blinkNr"])
                        if not pupil_data.loc["blinkNr"].item() == np.nan:

                            if pupil_data.loc["blinkNr"].item().is_integer():

                                if pupil_data.loc["blinkNr"].item() < 10:
                                    folder_blink_img = "images/blinks/000" + str(
                                        int(pupil_data.loc["blinkNr"].item())) + "/"
                                elif pupil_data.loc["blinkNr"].item() < 100:
                                    folder_blink_img = "images/blinks/00" + str(int(pupil_data.loc["blinkNr"].item())) + "/"
                                elif pupil_data.loc["blinkNr"].item() < 1000:
                                    folder_blink_img = "images/blinks/0" + str(int(pupil_data.loc["blinkNr"].item())) + "/"
                                else:
                                    folder_blink_img = "images/blinks/" + str(int(pupil_data.loc["blinkNr"].item())) + "/"

                                folder_blink_img = os.path.join(file_path, folder_blink_img)

                                if not os.path.exists(folder_blink_img):
                                    os.makedirs(folder_blink_img)

                                cv2.imwrite(folder_blink_img +"/" +  str(fn_eye_video) + "_" + str(frame_idx) + '.jpg', frame)

                    # # display video frame
                    if display_image_extraction:
                        cv2.imshow(fn_eye_video, frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                frame_idx += 1
            if not key == ord('q'):
                b_meta_data = pd.read_csv(os.path.join(file_path, 'blink_detection_report.csv'), index_col=0).T
                b_meta_data[str('image_extraction_eye' + str(eye_id))] = "completed"
                b_meta_data.T.to_csv(os.path.join(file_path, 'blink_detection_report.csv'))
                print("blink meta data change")
            video.release()
            cv2.destroyAllWindows()
        print("video dissection finalized")

        return

    def VideoFrameInfo(frame, pupil_data, frame_idx, pupil_time):
        confi = np.round(pupil_data.loc["confidence"].item(), decimals=3)
        pos_x = np.round(pupil_data.loc["norm_pos_x"].item(), decimals=4)
        pos_y = np.round(pupil_data.loc["norm_pos_y"].item(), decimals=4)
        pupil_dia = np.round(pupil_data.loc["diameter_3d"].item(), decimals=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "frame idx: " + str(frame_idx), (3, 20), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "timestamp: " + str(np.round(pupil_time, decimals=3)), (200, 20), font, .5, (255, 255, 255),
                    1, cv2.LINE_AA)
        cv2.putText(frame, "blink ID: ", (3, 40), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        if pupil_data.loc["blinkNr"].item().is_integer():
            cv2.putText(frame, str(int(pupil_data.loc["blinkNr"].item())), (70, 40), font, .5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.putText(frame, "confidence:     pos_x:     pos_y:    pupil dia:", (10, 355), font, .5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(frame, str(confi), (10, 375), font, .5, (0, confi * 255, (255 - confi * 255)), 1, cv2.LINE_AA)
        cv2.putText(frame, str(pos_x), (140, 375), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(pos_y), (230, 375), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(pupil_dia), (310, 375), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "mm", (350, 375), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def blink_detailes_record(blink_intime_matrix, eye_id, file_path):
        # Load blink data and flag in pupil_positions
        Bdata = pd.read_csv(os.path.join(file_path, 'blinks.csv'))
        # # read out blink index and and eye index out of file name of generated blink images
        # idx = int(file_name[file_name.find(".jpg") + -8:-8])
        idx_subfolder = int(subfolder[-4:]) - 1  # blink folder start with 1 and not 0
        # idx_Bdata_detail = 2 * idx + (int(file_name[file_name.find(".jpg") + -3:-6]))
        idx_Bdata_detail = 2 * idx_subfolder + eye_id
        # # get details and write to new Bdata_detail pd.dataframe
        Bdata_detail.loc[idx_Bdata_detail, "blink_id"] = Bdata.loc[idx_subfolder, "id"]
        Bdata_detail.loc[idx_Bdata_detail, "eye_id"] = eye_id
        Bdata_detail.loc[idx_Bdata_detail, "start_timestamp"] = Bdata.loc[idx_subfolder, "start_timestamp"]
        Bdata_detail.loc[idx_Bdata_detail, "duration"] = Bdata.loc[idx_subfolder, "duration"]
        Bdata_detail.loc[idx_Bdata_detail, "start_frame_index"] = Bdata.loc[idx_subfolder, "start_frame_index"]
        Bdata_detail.loc[idx_Bdata_detail, "index"] = Bdata.loc[idx_subfolder, "index"]
        Bdata_detail.loc[idx_Bdata_detail, "end_frame_index"] = Bdata.loc[idx_subfolder, "end_frame_index"]
        Bdata_detail.loc[idx_Bdata_detail, "confidence"] = Bdata.loc[idx_subfolder, "confidence"]
        Bdata_detail.loc[idx_Bdata_detail, "lid_blink_ypos_max"] = (np.nanmax(lid_blink_ypos) - upper_lid_ypos_avg)

        # set confidence value (element count of lid positions) // maybe as well the std???
        if upper_lid_ypos_elcount == 0 or bottom_lid_ypos_elcount == 0:
            amplitude_confidence = np.nan
        else:
            amplitude_confidence = np.round((m.sqrt(upper_lid_ypos_elcount) * m.sqrt(bottom_lid_ypos_elcount)) / 20,
                                            decimals=6)
        Bdata_detail.loc[idx_Bdata_detail, "amplitude_confidence"] = amplitude_confidence
        # calculate amplitude
        if amplitude_confidence == 0:
            Bdata_detail.loc[idx_Bdata_detail, "amplitude"] = np.nan
        else:
            Bdata_detail.loc[idx_Bdata_detail, "amplitude"] = (
                    (np.nanmax(lid_blink_ypos) - upper_lid_ypos_avg) / (bottom_lid_ypos_avg - upper_lid_ypos_avg))

        Bdata_detail.loc[idx_Bdata_detail, "upper_lid_ypos_avg"] = upper_lid_ypos_avg
        Bdata_detail.loc[idx_Bdata_detail, "upper_lid_ypos_std"] = upper_lid_ypos_std
        Bdata_detail.loc[idx_Bdata_detail, "upper_lid_ypos_elcount"] = upper_lid_ypos_elcount
        Bdata_detail.loc[idx_Bdata_detail, "bottom_lid_ypos_avg"] = bottom_lid_ypos_avg
        Bdata_detail.loc[idx_Bdata_detail, "bottom_lid_ypos_std"] = bottom_lid_ypos_std
        Bdata_detail.loc[idx_Bdata_detail, "bottom_lid_ypos_elcount"] = bottom_lid_ypos_elcount
        Bdata_detail.loc[idx_Bdata_detail, "lid_blink_ypos"] = lid_blink_ypos
        return Bdata_detail

# save disk space with removing extracted video images
save_disc_space_delete_eye_image_from_video = False

# set recording location
recordings_location = r'D:\recordings_Hard_drive'
demo_recording = '\\2019_06_12_SM34\\SM34_10mins_conversation'
export_source_path = 'exports\\002'

analyse_demo_recording = False
# extract blink related images for eye_ids and store in separate folders
extract_blink_images = True # just need to be done once

if analyse_demo_recording:
    file_path = os.path.join(recordings_location, demo_recording, export_source_path)
else:
    file_path = PLH.find_all_recordings(recordings_location)
# check if all source files are available and set available eye_ids

print((file_path))

if extract_blink_images:

    for fp in file_path:
        print(fp)
        eye_ids = PLH.check_source_f(fp)
        PLH.extract_blink_related_images(eye_ids, fp,
                                     display_image_extraction=False,  # visual control of extraction / faster without
                                     display_pupil_ellipse=False,  # visual control: pupil info and eye image matched
                                     # correctly  / take out for amplitude analysis
                                     display_pupil_details=False
                                     # writes pupil info at timestamp of eye image onto the eye image
                                     # / take out for amplitude analysis
                                     )

# Dataframe to record blink data "Bdata_detail"
column_names = ["blink_id",  # blink id from PL blinks.csv
                "eye_id",  # separating eye 0 and eye 1
                "start_timestamp",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "duration",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "start_frame_index",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "index",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "end_frame_index",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "confidence",  # from PL blinks.csv - not adjusted for eye 0 or eye0
                "amplitude",  # (bottom_lid_ypos_avg - upper_lid_ypos_avg) / (lid_blink_ypos_max - upper_lid_ypos_avg)
                "amplitude_confidence",  # measure of how many data has been taken into consideration
                "lid_blink_ypos_max",  # lid position max y position value during blink
                "upper_lid_ypos_avg",  # from 10 first and 10 last value of blink interval average position
                "upper_lid_ypos_std",  # std
                "upper_lid_ypos_elcount",  # element count of 20 possible
                "bottom_lid_ypos_avg",  # from 10 first and 10 last value of blink interval average position
                "bottom_lid_ypos_std",  # std
                "bottom_lid_ypos_elcount",  # element count of 20 possible
                "lid_blink_ypos"]  # all identified blink positions in generated blink images (green)
# in blink interval with actual low confidence
Bdata_detail = pd.DataFrame(columns=column_names)

# # extract just data from pupil center and stitch new blink image together
# # #find all blink image folders
blink_img_fp = os.path.join(file_path, 'images/blinks')
for eye_id in eye_ids:
    if eye_id == 0:
        image_file_ident = "/eye0.mp4_*.jpg"
        timestamp_fn = "/eye0_timestamps.npy"
    if eye_id == 1:
        image_file_ident = "/eye1.mp4_*.jpg"
        timestamp_fn = "/eye1_timestamps.npy"

    timestamps = np.load(os.path.abspath(os.path.join(file_path, "../..")) + timestamp_fn)
    timestamps = np.round(timestamps, decimals=6)
    pupil_positions = pd.read_csv(os.path.join(file_path,  "pupil_positions.csv"), index_col="pupil_timestamp")
    pupil_positions.index = np.round(pupil_positions.index, decimals=6)
    pupil_positions = pupil_positions[pupil_positions["eye_id"] == eye_id]
    # # find all subfolders for individual blink
    subfolders = [f.path for f in os.scandir(blink_img_fp) if f.is_dir()]
    for subfolder in subfolders:
        AllInOneImage = np.zeros((384, 1), np.uint8)
        if 'xpos_imaging' in locals() or 'xpos_imaging' in globals():
            del xpos_imaging

        # allImages_fn = sorted(glob.glob(subfolder + image_file_ident, recursive=True), key=os.path.getmtime)
        allImages_fn = sorted(glob.glob(subfolder + image_file_ident, recursive=True),
                              key=lambda x: int(x.split('\\')[-1].split('_')[-1][0:-4]))

        # store all found blink info in blink_intime_matrix
        # upper lid
        # bottom lid
        # top lid position with low confidence - > actual blink movement of lid
        blink_intime_matrix = np.zeros((3, len(allImages_fn)), dtype=float)
        for idx, file_name in enumerate(allImages_fn):
            # # eye_timestamp find eye data
            timestamp_id = int(file_name[file_name.find("mp4_") + 4:-4])
            pupil_time = timestamps[timestamp_id]
            pupil_data = pd.DataFrame(pupil_positions.loc[pupil_time])

            # # # xpos / ypos are coordinates of pupil centre
            # # # eye cam in OD is flipped so the direction of magnification factor need to be adjusted by. OD *-2 + 384
            if eye_id == 0:
                xpos = int(-2 * int(pupil_data.loc["ellipse_center_x"].item())) + 384
                ypos = int(-2 * int(pupil_data.loc["ellipse_center_y"].item())) + 384
            else:
                xpos = int(2 * int(pupil_data.loc["ellipse_center_x"].item()))
                ypos = int(2 * int(pupil_data.loc["ellipse_center_y"].item()))
            # two consecutive timestamps need confidence >0.8, initial set standard even when below 0.8
            confi = float(pupil_data.loc["confidence"].item())
            if 'xpos_imaging' not in locals() or 'xpos_imaging' not in globals():
                xpos_imaging = xpos
                confi_previous = confi
            # set high confidence values for solid pupil center display and lid detection
            if confi > 0.90:
                pupil_believable = True
                xpos_imaging = xpos
            elif (confi_previous >= 0.8) and (confi >= 0.8):
                pupil_believable = True
                xpos_imaging = xpos
            else:
                pupil_believable = False
                # print("low confidence values")

            eye_image = cv2.imread(file_name, 2)

            # image enhancement if not done previously
            eye_image = cv2.equalizeHist(eye_image)
            # do not override original image
            # no! cv2.imwrite(file_name, eye_image)
            # # possible image enhancement, but not as good as equalizeHist
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # eye_image = clahe.apply(eye_image)

            # slice just 10 rows out of the image around the center
            # eye_image_binary =
            eye_image = eye_image[0:384, xpos_imaging - 5:xpos_imaging + 5]
            # # # average 10 image points
            average_image_pc = (np.mean(eye_image, axis=1))

            # # # find lid positions upper and bottom

            if pupil_believable and ypos < 384:
                # draw vertical position of pupil onto image
                eye_image[ypos, :] = 255
                # # bottom lid identification
                # find_peak does not work good enough
                lower_Lid_max_pos = ypos + 130
                if lower_Lid_max_pos >= 384:
                    lower_Lid_max_pos = 384
                find_bottom_lid_array = average_image_pc[ypos + 80:lower_Lid_max_pos] - \
                                        average_image_pc[ypos + 79:lower_Lid_max_pos - 1]
                peak_indices1 = np.argmax(find_bottom_lid_array)
                bottom_lid_pos = (peak_indices1 + ypos + 79)
                if bottom_lid_pos < 384:
                    eye_image[bottom_lid_pos, :] = 255

                # # # top lid detection
                find_upper_lid_array = average_image_pc[ypos - 90:ypos - 50]
                find_upper_lid_array2 = average_image_pc[ypos - 90:ypos - 50] - average_image_pc[ypos - 91:ypos - 51]
                peak_indices2 = np.argmax(-find_upper_lid_array)
                upper_lid_pos = (peak_indices2 + ypos - 90)
                # plt.scatter(peak_indices1, find_upper_lid_array2[peak_indices1])
                if upper_lid_pos > 0:
                    eye_image[upper_lid_pos, :] = 255


                blink_intime_matrix[0][idx] = upper_lid_pos
                blink_intime_matrix[1][idx] = bottom_lid_pos

            # # # check lid movements during blink - when pupil detection has low confidence
            if not pupil_believable:
                find_lid_drop_array = average_image_pc[0:384]
                # find first min in find_lid_drop_array
                peaks, _ = find_peaks(find_lid_drop_array, distance=30, height=100, prominence=20)
                peaks_min, _ = find_peaks(255 - find_lid_drop_array[peaks[0]:], distance=30, height=100, prominence=15)

                print("peaks[0]: ", peaks[0])
                print("peaks_min[0]: ", peaks_min)

                if peaks[0] and peaks_min.any():
                    peaks_min_idx = peaks_min[0] + peaks[0]
                    # find value .8 of difference between min and max value
                    threshold = .2 * (find_lid_drop_array[peaks[0]] - find_lid_drop_array[peaks_min_idx]) + \
                                find_lid_drop_array[peaks_min_idx]
                    print(np.argmin(abs(find_lid_drop_array[peaks[0]:peaks_min_idx] - threshold)) + peaks[0])

                    lid_posY = np.argmin(abs(find_lid_drop_array[peaks[0]:peaks_min_idx] - threshold)) + peaks[0]
                    blink_intime_matrix[2][idx] = lid_posY

                    # print("threshold: ", threshold)

                    print("find_lid_drop_array[peaks[0]]: ", find_lid_drop_array[peaks[0]])
                    print("find_lid_drop_array[peaks_min[0]]: ", find_lid_drop_array[peaks_min[0]])


                #     plt.plot(lid_posY, find_lid_drop_array[lid_posY], "o", c='r')
                # plt.plot(find_lid_drop_array)
                # # plt.plot(255 - find_lid_drop_array)
                # plt.plot(peaks[0], find_lid_drop_array[peaks[0]], "x")
                #
                # plt.plot(peaks_min_idx, find_lid_drop_array[peaks_min_idx], "x")
                #
                # plt.title(subfolder.split('\\')[-2] + "/" + file_name.split('\\')[-1])
                # plt.show()


                # # find min in find_lid_drop_array
                # search_array_max_idx = np.argmin(find_lid_drop_array[50:]) + 50
                # search_array_max_idx_value = find_lid_drop_array[search_array_max_idx]
                # # find max in find_lid_drop_array
                # search_array_min_idx = np.argmax(find_lid_drop_array[:search_array_max_idx])
                # search_array_min_idx_value = find_lid_drop_array[search_array_min_idx]
                # # find value that drops
                # delta_max_min_value = (search_array_min_idx_value - search_array_max_idx_value)
                # if (search_array_min_idx_value - search_array_max_idx_value) < 180:
                #     delta_max_min_value = delta_max_min_value * .50 + search_array_max_idx_value
                # else:
                #     delta_max_min_value = delta_max_min_value * 0.4 + search_array_max_idx_value
                #
                # lid_posY = np.argmin(abs(find_lid_drop_array[search_array_min_idx:search_array_max_idx] - (
                #     delta_max_min_value))) + search_array_min_idx
                #
                # blink_intime_matrix[2][idx] = lid_posY
                # # display for analyses
                # plt.plot(find_lid_drop_array)
                # # plt.plot(find_upper_lid_array2)
                # plt.scatter(lid_posY, find_lid_drop_array[lid_posY])
                # plt.scatter(search_array_max_idx, search_array_max_idx_value)
                # plt.scatter(search_array_min_idx, search_array_min_idx_value)
                # plt.title(subfolder.split('\\')[-2] + "/" +  file_name.split('\\')[-1])
                # plt.show()
            confi_previous = confi
            # stitch all image slices together
            AllInOneImage = np.concatenate((AllInOneImage, eye_image), axis=1)

        blink_intime_matrix[blink_intime_matrix < 0.1] = np.nan
        # AllInOneImage = cv2.equalizeHist(AllInOneImage)
        AllInOneImage = cv2.merge([AllInOneImage, AllInOneImage, AllInOneImage])

        # record (blink_intime_matrix)
        upper_lid_ypos_avg = np.nanmean([blink_intime_matrix[0][:10], blink_intime_matrix[0][-10:]])
        upper_lid_ypos_std = np.nanstd([blink_intime_matrix[0][:10], blink_intime_matrix[0][-10:]])
        upper_lid_ypos_elcount = sum(sum(~np.isnan([blink_intime_matrix[0][:10], blink_intime_matrix[0][-10:]])))
        if not np.isnan(upper_lid_ypos_avg):
            AllInOneImage[int(upper_lid_ypos_avg), :, [0, 1]] = 0
            AllInOneImage[int(upper_lid_ypos_avg), :, 2] = 255
        bottom_lid_ypos_avg = np.nanmean([blink_intime_matrix[1][:10], blink_intime_matrix[1][-10:]])
        bottom_lid_ypos_std = np.nanstd([blink_intime_matrix[1][:10], blink_intime_matrix[1][-10:]])
        bottom_lid_ypos_elcount = sum(sum(~np.isnan([blink_intime_matrix[1][:10], blink_intime_matrix[1][-10:]])))

        lid_blink_ypos = blink_intime_matrix[2]
        # visualise data in blink image
        if not np.isnan(bottom_lid_ypos_avg):
            AllInOneImage[int(bottom_lid_ypos_avg), :, [0, 1]] = 0
            AllInOneImage[int(bottom_lid_ypos_avg), :, 2] = 255
        for idx, value in enumerate(blink_intime_matrix[2]):
            if not np.isnan(value):
                value = int(value)
                AllInOneImage[value, idx * 10 + 1:idx * 10 + 11, [0, 2]] = 0
                AllInOneImage[value, idx * 10 + 1:idx * 10 + 11, 1] = 255
        # save as one image
        img_out_fn = blink_img_fp + "/" + str(subfolder[-4:]) + "_" + str(eye_id) + 'hs.jpg'
        cv2.imwrite(img_out_fn, AllInOneImage)
        # record found data in (Bdata_detail)
        PLH.blink_detailes_record(blink_intime_matrix, eye_id, file_path)

print(Bdata_detail["amplitude"])
Bdata_detail.to_csv(r'blinks_detailed.csv', index=False)
print("Blink amplitude images and data file generated")

# if available, display image eye0 in top of image eye1
if eye_ids.shape[0] > 1:
    image_file_ident = "/*hs.jpg"
    # # find all blink images
    allImages_fn = sorted(glob.glob(blink_img_fp + image_file_ident, recursive=True))
    # # find all images belonging to eye0 and eye1
    allImages_fn0 = list(filter(lambda x: x[-7] == "0", allImages_fn))
    allImages_fn1 = list(filter(lambda x: x[-7] == "1", allImages_fn))
    # # stitch 1 & 2 together in top of each other
    for idx, Image_fn0 in enumerate(allImages_fn0):
        eye_image0 = cv2.imread(allImages_fn0[idx])
        eye_image1 = cv2.imread(allImages_fn1[idx])
        # # check which image width is smaller and amend to length of other.
        width_eye_0 = eye_image0.shape[1]
        width_eye_1 = eye_image1.shape[1]
        if width_eye_0 < width_eye_1:
            image_zeros = np.zeros(eye_image1.shape)
            image_zeros [:,0:width_eye_0,:] = eye_image0
            eye_image0 = image_zeros
        if width_eye_1 < width_eye_0:
            image_zeros = np.zeros(eye_image0.shape)
            image_zeros[:, 0:width_eye_1, :] = eye_image1
            eye_image1 = image_zeros
        AllInOneImage = np.concatenate((eye_image0, eye_image1), axis=0)
        img_out_fn = blink_img_fp + "/binocular_blink_image_" + str(idx + 1) + '.jpg'
        cv2.imwrite(img_out_fn, AllInOneImage)
    for image_fn in allImages_fn:
        os.remove(image_fn)

# save disc space
if save_disc_space_delete_eye_image_from_video:
    subfolders = [f.path for f in os.scandir(blink_img_fp) if f.is_dir()]
    for subfolder in subfolders:
        shutil.rmtree(subfolder)
    print("excessive images deleted")
print("Data extraction finalised")
