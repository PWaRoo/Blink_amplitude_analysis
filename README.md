# Blink_amplitude_analysis

required sources from pupil labs eye-tracking data: 
 - eye videos (original) 
 - pupil_positions.csv
 - blinks.csv 

Output: 
  - blinks_detaliled.csv based on blinks.csv but for both eyes seperately 
 
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
