
# Summary
Contains the three iterations (largely full rewrites) of this code, roughly by year. (I was not the greatest at version control back then). The code is all unchanged from back then, besides making it easy to run an example.   
Almost all of the images/dicoms have been removed, a few redacted ones are left in to show what the programs steps (in ```example_images```)     

# Problem:
A company (left unnamed publicly), has a large number of scanned (often poorly), handwritten paitent history forms. There is now need (for training AI models) to gather paitents that 1. Have not had prior breast cancer and 2. Have not had prior breast surgery. You can either hire a human who will quit after a day because of how boring it is or a motivated highschooler hears about it at dinner one day. 

There are some example forms in the root of the repo, first thing is that the data is not even all the same or even correct type of form, which takes away the easy approach of looking 
in the same spot for a check. 

# Process:
You can see my testing, espically in 2021 where I didn't know how to do any of the needed steps (reading dicom, converting to pdf, anything with openCV, etc). By 2023 though there was nice logging, and the overall program was much simpler since I actually knew how to solve the problem.  

# Solution
I use OCR (```pytesseract```) to essentially "orient" within the page. It looks for the relevent text "has had cancer/surgery". If it can't find it, it's an easy toss out. The next step is finding all the possible checkboxes, which the Python library ```boxdetect``` does. Then we look for a section of 4 boxes that are in line with the detected text,   


# How to try
Install UV https://docs.astral.sh/uv/getting-started/installation/

Clone, run ```uv sync```, then ```cd y2023``` and ```uv run main.py -i FastPath --debug```. You will see a debug image show up that annotating what the program found (i.e boxes and text), it will also produce logs and an example .csv out. 