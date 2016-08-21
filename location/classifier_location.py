#-*-coding=utf-8 -*-
# 东北：辽宁,吉林,黑龙江
# 华北：河北,山西,内蒙古,北京,天津
# 华东：山东,江苏,安徽,浙江,台湾,福建,江西,上海
# 华中：河南,湖北,湖南
# 华南：广东,广西,海南,香港,澳门
# 西南：云南,重庆,贵州,四川,西藏
# 西北：新疆,陕西,宁夏,青海,甘肃
# 境外：其他

if __name__ == '__main__':
	contentFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_status.txt"
	labelsFile = "/mnt/sdb/SMP_CUP_2016/train/train/train_labels.txt"
	
	northeastFile = "./northeast"
	northChinaFile = "./northChina"
	eastChinaFile = "./eastChina"
	centralChinaFile = "./centralChina"
	southChinaFile = "./southChina"
	southwestFile = "./southwest"
	northwestFile = "./northwest"
	abroadFile = "./abroad"

	finContent = open(contentFile, "r")
	finLabels = open(labelsFile, 'r')
	finNortheast = open(northeastFile, 'w')
	finNorthChina = open(northChinaFile, 'w')
	finEastChina = open(eastChinaFile, 'w')
	finCentralChina = open(centralChinaFile, 'w')
	finSouthChina = open(southChinaFile, 'w')
	finSouthwest = open(southwestFile, 'w')
	finNorthwest = open(northwestFile, 'w')
	finAbroad = open(abroadFile, 'w')

	for lineContent in finContent:
		uid = lineContent.split(',')[0]
		finLabels.seek(0)
		for lineInfo in finLabels:
			if (uid == lineInfo.split('||')[0]):
				content = lineContent.split(',')[5]
				place = lineInfo.split('||')[3].split(' ')[0]
				if (place == "辽宁" or place == "吉林" or place == "黑龙江"):
					finNortheast.write(content)
					break
				elif (place == "河北" or place == "山西" or place == "内蒙古" or place == "北京" or place == "天津"):
					finNorthChina.write(content)
					break
				elif (place == "山东" or place == "江苏" or place == "安徽" or place == "浙江" or place == "台湾" or place == "福建" or place == "江西" or place == "上海"):
					finEastChina.write(content)
					break
				elif (place == "河南" or place == "湖北" or place == "湖南"):
					finCentralChina.write(content)
					break
				elif (place == "广东" or place == "广西" or place == "海南" or place == "香港" or place == "澳门"):
					finSouthChina.write(content)
					break
				elif (place == "云南" or place == "重庆" or place == "贵州" or place == "四川" or place == "西藏"):
					finSouthwest.write(content)
					break
				elif (place == "新疆" or place == "陕西" or place == "宁夏" or place == "青海" or place == "甘肃"):
					finNorthwest.write(content)
					break
				elif (place != "None"):
					finAbroad.write(content)
					break
	finContent.close()
	finLabels.close()
	finNortheast.close()
	finNorthChina.close()
	finCentralChina.close()
	finEastChina.close()
	finNorthwest.close()
	finSouthwest.close()
	finSouthChina.close()
	finAbroad.close()
