Author: Christoper Holley
Email: torchtopher@gmail.com

# History Sheet Scanner

Installation:
Make sure python 3 is installed.
run "pip install -r requirements.txt" if there are errors try "pip3 install -r requirements.txt"
Hopefully all packages needed are now installed.

Usage:
Directory Structure:
    -Script Files
    -Directory with Dicom Files
    -testset
    -poppler-0.68.0_x86
    -Other files

To test, run "python NewHistoryScan.py testset"
Check the resulting HXout.csv and to see if it worked. There should be 10 entries.

If it completes with no errors then it should work on the larger directory.
run "python NewHistoryScan.py <Directory with Dicom Files>"

If you want to change the output name from default (HXout.csv) you can do something like this using the --o flag:
"python NewHistoryScan.py <Directory with Dicom Files> --o MyoutputName"

Something will have probably gone wrong but if not it should start evaluating the files in the directory you provided.
It will also produce some temporary files that can be deleted but will just be made again on next run.


What each field means:
	FileName: Pretty simple, just the name of the file that was scanned, only used as a way for the program to keep track of what has already been scanned.
	Result: If the file can be used or not, if it is a 1 it means that form was a history form and had the correct boxes checked.
	Cancer/Surgery: If either of these are 1 it means that the form was a history form but they had cancer/surgery.
	DicomReadFail: If the program can not open the .dcm file or there is no document inside of it.
	BoxesFailed vs NotHX: Probably the most important difference. BoxesFailed means that the program thinks it might be a history form, but was unable to be sure where the detected boxes on the page were.
				NotHX means it detected less than 4 boxes. This means that the file is most likely not a history form, or at the very least not a form it could read. To put it simply, BoxFailed means that the program did not have enough boxes to determine
				where they were on the page and NotHX means the number of boxes detected was less than 4.
	Patient_ID,DOB,Study_Instance_UID,StudyDate: All what they sound like. The program does not sort by these and check for the most recent history form, but that can be done after the output is generated if needed. If the inputs are only the most recent forms for the paitent
						then it will not be a problem.
	MultiCheckBoxes: Means that it is a history form and that there were more or less than 2 boxes.

Possible ways to use the output:
	Assuming the inputs are only the most recent forms, then just sort by Patient_ID and DOB and look for a 1 in the results column. Keep track of the Study_Instance_UID or whatever could be used to grab the actual images.

	If the inputs are just all the forms for all the patients then it will be a bit harder.
		I can write this program if needed.
		Assume that people have 1 or less history forms a year. Assume that their will be many other files that are not history forms in the same year.
		For each patient, determined by Patient_ID and DOB, use study date to determine the most recent year that the patient has forms for.
		Then check all those forms and if there is a 1 in the results column then you know that the person can be used.
		This is because a "Positive" result in the most recent year and knowing they only have 1 form per year means they can not have had cancer/surgery in any other year.




