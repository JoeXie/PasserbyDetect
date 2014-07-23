#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// function declaration
void deleteBackground(Mat& srcImage, Mat& model);


int main(int argc, char **argv)
{
	// 1. ������ͼ���Ҫ�����ͼ��
	Mat sampleImage = imread("E:\\SmartCity\\���ݼ�/0004011.png");
	char* imagePrepath = "E:\\SmartCity\\���ݼ�\\��֤����\\1_2_01_1\\hongsilounorth_13_1920x1080_30_R1\\";
	char* resultPrepath = "E:\\SmartCity\\���ݼ�\\������\\";

	if(sampleImage.data == NULL)
	{
		cout << "ģ��ͼƬ�򿪴���"<< endl;
		return false;
	}
	
	char* namelistPath = new char[200];
	strcpy(namelistPath, imagePrepath);
	strcat(namelistPath, "Namelist.txt");
	FILE* namelistIn = fopen(namelistPath, "r");  //���ļ������룬���ط���ָ��FILE�����ָ��
	if(namelistIn == NULL)
	{
		printf("Can not open the namelist."); 
		return false;
	}

	char* imageName = new char [100]; //ͼ���ļ���
	while(fscanf(namelistIn, "%s", imageName) > 0) //��ȡ�ļ�������whitespaceֹͣ, ���ض�ȡ���ַ�����
	{
		printf("\n");


		// 2. ��ͼ��ĵ�һ�����ؽ��д���
		char* imagePath = new char[200];
		strcpy(imagePath, imagePrepath);
		strcat(imagePath, imageName); //���ڶ����ַ���ƴ�ӵ���һ������
		printf("Open image: %s\n",imageName); //��ӡ���ļ���
		Mat image = imread(imagePath);
		if(image.data == NULL)
		{
			printf("��ͼƬʧ�ܣ�\n");
			return false;
		}

		deleteBackground(image, sampleImage);

		// 3. ����ͼ��
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