#-*-coding=utf-8 -*-

if __name__ == '__main__':
	contentFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_status.txt"
	labelsFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_labels.txt"
	maleFile = "./male"
	femaleFile = "./female"

	finContent = open(contentFile, 'r')
	finLabels = open(labelsFile, 'r')
	finMale = open(maleFile, 'w')
	finFemale = open(femaleFile, 'w')
	for lineContent in finContent :
		uid = lineContent.split(',')[0]
		finLabels.seek(0)
		for lineInfo in finLabels :
			if(uid == lineInfo.split('||')[0]) :
				content = lineContent.split(',')[5]
				if(lineInfo.split('||')[1] == 'm') :
					finMale.write(content)
					break
				elif(lineInfo.split('||')[1] == 'f') :
					finFemale.write(content)
					break
	finContent.close()
	finLabels.close()
	finMale.close()
	finFemale.close()
