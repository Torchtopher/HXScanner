
# Summary
Contains the three iterations (largely full rewrites) of this code, roughly by year. (I was not the greatest at version control back then.) The code is all unchanged from back then, besides making it easy to run an example.   
Almost all of the images/dicoms have been removed, a few redacted ones are left in to show what the programs steps (in ```example_images```)     

# Problem:
An imaging company had a large number of (often poorly) scanned handwritten patient history forms which they needed to extract data from to train AI models. The AI needed patients who had no history of breast cancer or prior breast surgery. You can either hire a human who will quit after a day because of how boring it is or a motivated highschooler writes a program to extract the data :). The big takeaway is that the data is not all the same or even correct type of form, which takes away the easy approach of looking in the same spot for a filled in checkbox.  

There are some example forms in ```example_images```. I've included annotated images that visually show the process (i.e finding the text, finding the checkboxes and figuring out top left, top right etc). 

# Process:
You can see my testing, espically in 2021 when I didn't know how to do any of the needed steps (reading dicom, converting to pdf, anything with openCV, etc). By 2023 I had the overall program was much simpler since I actually knew how to solve the problem.  

# Solution
I use OCR (```pytesseract```) to essentially "orient" within the page. It looks for the relevent text "has had cancer/surgery". If it can't find it, it's an easy toss out. The next step is finding all the possible checkboxes, which the Python library ```boxdetect``` does. Then we look for a section of 4 boxes that are in line with the detected text, (i.e close enough Y value). After the boxes are determined it just counts the number of pixels that are darker than a threshold. The program is happy to throw out anything that doesn't fit perfectly, since precision was the primary concern over recall.      


# How to try
Install UV https://docs.astral.sh/uv/getting-started/installation/

Clone, run ```uv sync``` in the root, then ```cd y2023``` and ```uv run main.py -i FastPath --debug```. You will see a debug image show up that annotating what the program found (i.e boxes and text), and after hitting Q it will save the image to y2023/TestAlgo and produce logs and an example .csv out.