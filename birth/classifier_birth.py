#-*-coding=utf-8 -*-
#-1979/1980-1989/1990+

if __name__ == '__main__':
	contentFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_status.txt"
	labelsFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_labels.txt"
	oldFile = "./_1979"
	normalFile = "./1980-1989"
	youngFile = "./1990+"

	finContent = open(contentFile, 'r')
	finLabels = open(labelsFile, 'r')
	finOld = open(oldFile, 'w')
	finNormal = open(normalFile, 'w')
	finYoung = open(youngFile, 'w')

	for lineContent in finContent :
		uid = lineContent.split(',')[0]
		finLabels.seek(0)
		for lineInfo in finLabels :
			if(uid == lineInfo.split('||')[0]) :
				content = lineContent.split(',')[5]
				if(lineInfo.split('||')[2] <= '1979') :
					finOld.write(content)
					break
				elif(lineInfo.split('||')[2] <= '1989') :
					finNormal.write(content)
					break
				elif(lineInfo.split('||')[2] >= '1990') :
					finYoung.write(content)
					break
	finContent.close()
	finLabels.close()
	finOld.close()
	finNormal.close()
	finYoung.close()