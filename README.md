# Blink_amplitude_analysis

required sources from pupil labs eye-tracking data: 
 - eye videos (original) 
 - pupil_positions.csv
 - blinks.csv 

Output: 
  - blinks_detaliled.csv based on blinks.csv but for both eyes seperately 
  
     "blink_id",                # blink id from PL blinks.csv
     "eye_id",                  # separating eye 0 and eye 1
     "start_timestamp",         # from PL blinks.csv - not adjusted for eye 0 or eye0
     "duration",                # from PL blinks.csv - not adjusted for eye 0 or eye0
     "start_frame_index",       # from PL blinks.csv - not adjusted for eye 0 or eye0
     "index",                   # from PL blinks.csv - not adjusted for eye 0 or eye0
     "end_frame_index",         # from PL blinks.csv - not adjusted for eye 0 or eye0
     "confidence",              # from PL blinks.csv - not adjusted for eye 0 or eye0
     "amplitude",               # (bottom_lid_ypos_avg - upper_lid_ypos_avg) / (lid_blink_ypos_max - upper_lid_ypos_avg)
     "amplitude_confidence",    # measure of how many data has been taken into consideration
     "lid_blink_ypos_max",      # lid position max y position value during blink
     "upper_lid_ypos_avg",      # from 10 first and 10 last value of blink interval average position
     "upper_lid_ypos_std",      # std
     "upper_lid_ypos_elcount",  # element count of 20 possible
     "bottom_lid_ypos_avg",     # from 10 first and 10 last value of blink interval average position
     "bottom_lid_ypos_std",     # std
     "bottom_lid_ypos_elcount", # element count of 20 possible
     "lid_blink_ypos"]          # all identified blink positions in generated blink images (green)

Process: 
- flags blink related timestamps in pupil_positions.csv 
- extracts flaged eye images out of eye-videos
- assembles blink images to show slices of eye images over time 
- analyses each slice for blink related information
    - position of upper lid 
    - position of bottom lid 
    - positon of upper during blink
 
 envisioned enhancements: pre analyses of extracted blink data 
  - optimised values for onset/ offset confidence threshold and filter length 
  conditions: - check for blink duration - more than 1.5-2 sec is unbelivable 
  
  manually best value for onset 0.45 offset 0.35 filter length ?? 
  
  - from the top lid position during blink - the blink duration could possibly be detected 
  
cheers 
