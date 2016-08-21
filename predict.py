#-*-coding=utf-8 -*-
import numpy as np 

if __name__ == '__main__':
    labelPath = "./test_nolabels.txt"
    genderPath = "./gender/genderPredict"
    birthPath = "./birth/birthPredict"
    locationPath = "./location/locationPredict"

    predictPath = "./temp.csv"

    finLabel = open(labelPath, 'r')
    finGender = open(genderPath, 'r')
    finBirth = open(birthPath, 'r')
    finLocation = open(locationPath, 'r')
    finPredict = open(predictPath, 'w')


    finPredict.write("uid,age,gender,province"+'\n')

    sex = [3.0, 1.0]
    birth = [2.0, 4.0, 1.0]
    location = [1.0, 3.0, 10.0, 22.0, 10.0, 35.0, 35.0, 4.0]

    for lineLabel in finLabel:
        uid = lineLabel.strip()

        birthPridict = [0, 0, 0]
        finBirth.seek(0)
        for lineBirth in finBirth:
            line = lineBirth.strip().split(' ')
            if(line[0] == uid):
                predict = int(float(line[1]))
                birthPridict[predict] += 1
        for i in xrange(len(birth)):
            birthPridict[i] /= birth[i]
        if(birthPridict.index(max(birthPridict)) == 2):
             age = "-1979"
        elif(birthPridict.index(max(birthPridict)) == 1):
             age = "1980-1989"
        else:
             age = "1990+"

        genderPridict = [0, 0]
        finGender.seek(0)
        for lineGender in finGender:
            line = lineGender.strip().split(' ')
            if (line[0] == uid):
                predict = int(float(line[1]))
                genderPridict[predict] += 1
        for i in xrange(len(sex)):
            genderPridict[i] /= sex[i]
        if (genderPridict.index(max(genderPridict)) == 1):
            gender = "f"
        else:
            gender = "m"

        locationPridict = [0, 0, 0, 0, 0, 0, 0, 0]
        finLocation.seek(0)
        for lineLocation in finLocation:
            line = lineLocation.strip().split(' ')
            if (line[0] == uid):
                predict = int(float(line[1]))
                locationPridict[predict] += 1
        for i in xrange(len(location)):
            locationPridict[i] /= location[i]
        if (locationPridict.index(max(locationPridict)) == 7):
            province = "东北"
        elif (locationPridict.index(max(locationPridict)) == 6):
            province = "华北"
        elif (locationPridict.index(max(locationPridict)) == 5):
            province = "华东"
        elif (locationPridict.index(max(locationPridict)) == 4):
            province = "华中"
        elif (locationPridict.index(max(locationPridict)) == 3):
            province = "华南"
        elif (locationPridict.index(max(locationPridict)) == 2):
            province = "西南"
        elif (locationPridict.index(max(locationPridict)) == 1):
            province = "西北"
        else:
            province = "境外"

        finPredict.write(uid + ',' + age + ',' + gender + ',' + province + '\n')
        
    finLabel.close()
    finGender.close()
    finBirth.close()
    finLocation.close()
    finPredict.close()
        

