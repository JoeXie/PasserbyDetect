#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// function declaration
void deleteBackground(Mat& srcImage, Mat& model);


int main(int argc, char **argv)
{
	// 1. 打开样本图像和要处理的图像
	Mat sampleImage = imread("E:\\SmartCity\\数据集/0004011.png");
	char* imagePrepath = "E:\\SmartCity\\数据集\\验证数据\\1_2_01_1\\hongsilounorth_13_1920x1080_30_R1\\";
	char* resultPrepath = "E:\\SmartCity\\数据集\\处理背景\\";

	if(sampleImage.data == NULL)
	{
		cout << "模板图片打开错误！"<< endl;
		return false;
	}
	
	char* namelistPath = new char[200];
	strcpy(namelistPath, imagePrepath);
	strcat(namelistPath, "Namelist.txt");
	FILE* namelistIn = fopen(namelistPath, "r");  //打开文件，读入，返回返回指向FILE对象的指针
	if(namelistIn == NULL)
	{
		printf("Can not open the namelist."); 
		return false;
	}

	char* imageName = new char [100]; //图像文件名
	while(fscanf(namelistIn, "%s", imageName) > 0) //读取文件，遇到whitespace停止, 返回读取的字符个数
	{
		printf("\n");


		// 2. 对图像的第一个像素进行处理
		char* imagePath = new char[200];
		strcpy(imagePath, imagePrepath);
		strcat(imagePath, imageName); //将第二个字符串拼接到第一个后面
		printf("Open image: %s\n",imageName); //打印出文件名
		Mat image = imread(imagePath);
		if(image.data == NULL)
		{
			printf("打开图片失败！\n");
			return false;
		}

		deleteBackground(image, sampleImage);

		// 3. 保存图像
		char* resultPath = new char[200];
		strcpy(resultPath, resultPrepath);
		strcat(resultPath, imageName);
		imwrite(resultPath, image);

		delete[] imagePath;
		delete[] resultPath;
	}

	delete[] namelistPath;
	delete[] imageName;
	fclose(namelistIn);

	return true;
}


void deleteBackground(Mat& srcImage, Mat& model)
{
	for(int y = 0; y < model.rows; y++)
	{
		for(int x = 0; x < model.cols; x++)
		{

			if(model.at<Vec3b>(y, x)[0] < 3)
				for(int i = 0; i < 3; i++)
					srcImage.at<Vec3b>(y, x)[i] = 0;
		}
	}
}