﻿# Attendance-System

This is a repo to keep track of and make changes to the attendance-system/facial recognition system in project Vulcan of Electronics and Robotics Club, BITS Pilani K. K. Birla Goa Campus.

I have used the deepface library as a means to run facial recognition. However, I felt that the system was not always accurate and reliable as due to changes in lighting, environment and some other factors, deepface library made inaccurate predictions.
To resolve this issue, I tried to make a database for each person wherein multiple images of the person are kept so run the verification program on multiple instances of the same face so as to increase the probability of getting better results.

Moreover, the code also updates the database with new images of a person it detects as long as a set threshold is not exceeded, while creating new folders for new people it encounters.
