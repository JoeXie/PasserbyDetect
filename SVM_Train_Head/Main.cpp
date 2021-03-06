#include <opencv2/opencv.hpp>
#include "Mysvm.h"
#include "MyRect.h"

#include <ml.h>  
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>
#include <time.h>

using namespace std;
using namespace cv;

//函数声明
int train(char* positivePath, int positiveSampleCount, 
		  char* negativePath, int negativeSampleCount, char* classifierSavePath);
void saveDetectResult(Mat& img, CvRect rect, char* savePath);
vector<float> loadSVMDetector(char* path);
Mat& detect(Mat& img, vector<float> SVMDetector);


int main(int argc, char* argv[])
{

	char* posPrePath = "E:\\SmartCity\\HeadDetection\\Pos\\32x32\\";
	char* negPrePath = "E:\\SmartCity\\HeadDetection\\Neg\\negdata\\32x32\\";
	char* trainResultPrePath = "E:\\SmartCity\\HeadDetection\\Result\\";

	char* imagePrePath = "E:\\SmartCity\\HeadDetection\\Data\\";
	char* resultPrePath = "E:\\SmartCity\\HeadDetection\\Result\\";
	char* SVMDetectorPath = "E:\\SmartCity\\HeadDetection\\Result\\\\SVMDetector.txt";


	if( 1 ) //设置要不要训练
	{
		cout << "开始训练" << endl;
		int trainFlag = train(posPrePath, 70, negPrePath, 700, trainResultPrePath);
	}

	if( 1 ) //设置要不要检测
	{
		vector<float> SVMDetector = loadSVMDetector(SVMDetectorPath);

		char* namelistPath = new char[200];
		strcpy(namelistPath, imagePrePath);
		strcat(namelistPath, "Namelist.txt");


		FILE* namelistIn = fopen(namelistPath, "r");  //打开文件，读入，返回返回指向FILE对象的指针
		if(namelistIn == NULL)
		{
			printf("Can not open namelist."); 
			return 0;
		}

		char* imageName = new char [100]; //图像文件名
		while(fscanf(namelistIn, "%s", imageName) > 0) //读取文件，遇到whitespace停止, 返回读取的字符个数
		{
			printf("\n");

			char* imagePath = new char[200];
			strcpy(imagePath, imagePrePath);
			strcat(imagePath, imageName); //将第二个字符串拼接到第一个后面
			printf("Open image: %s\n",imageName); //打印出完整文件名

			Mat srcImage = cvLoadImage(imagePath);
			if(srcImage.data == NULL)
			{
				printf("没有图片\n");
				system("pause");
				return -1;
			}

			//缩放
			float scale = 1.2;
			Mat dstImage;
			resize(srcImage, dstImage, Size(0, 0), scale, scale, INTER_LINEAR); //双线性插值 

			dstImage = detect(dstImage, SVMDetector);

			char* resultPath = new char[200];
			strcpy(resultPath, resultPrePath);
			strcat(resultPath, imageName);
			imwrite(resultPath, dstImage);

			delete[] imagePath;
			delete[] resultPath;
		}

		delete[] namelistPath;
		delete[] imageName;
		fclose(namelistIn);

	}

	return 1;
}

int train(char* positivePath, int positiveSampleCount, 
	char* negativePath, int negativeSampleCount, char* classifierSavePath)
{
	time_t startTime = time(NULL);	//记录开始时间

	int totalSampleCount = positiveSampleCount + negativeSampleCount; //总的样本数
	cout<<"//////////////////////////////////////////////////////////////////"<<endl;
	cout<<"totalSampleCount: "<<totalSampleCount<<endl;
	cout<<"positiveSampleCount: "<<positiveSampleCount<<endl;
	cout<<"negativeSampleCount: "<<negativeSampleCount<<endl;

	CvMat *sampleFeaturesMat = cvCreateMat(totalSampleCount , 324, CV_32FC1);
	//64*128的训练样本，该矩阵将是totalSample*3780,64*64的训练样本，该矩阵将是totalSample*1764
	cvSetZero(sampleFeaturesMat);  
	CvMat *sampleLabelMat = cvCreateMat(totalSampleCount, 1, CV_32FC1);//样本标识  
	cvSetZero(sampleLabelMat);  


	cout<<"************************************************************"<<endl;
	cout<<"start to training positive samples."<<endl;

	char* positiveNamelistPath = new char[200]; //文本文档完整文件路径
	strcpy(positiveNamelistPath, positivePath);
	strcat(positiveNamelistPath, "Namelist.txt"); //产生图像文件名文本文档的完整路径名

	FILE* fileIn = fopen(positiveNamelistPath, "r");  //打开文件，读入，返回返回指向FILE对象的指针
	if(fileIn == NULL)
	{
		printf("Can not open file."); 
		return 0;
	}

	delete[] positiveNamelistPath; //释放内存

	for(int i=0; i<positiveSampleCount; i++)  
	{  
		char* positiveImageName = new char[100]; //图像文件名
		fscanf(fileIn,"%s",positiveImageName); //读取文件，遇到whitespace停止
		char* positiveImagePath = new char[200]; //存储完整文件路径
		strcpy(positiveImagePath, positivePath);
		strcat(positiveImagePath,positiveImageName); //将第二个字符串拼接到第一个后面
		printf("%s\n",positiveImagePath); //打印出完整文件路径名

		Mat img = cv::imread(positiveImagePath);
		if(img.data == NULL)
		{
			cout<<"Positive image sample load error: "<<positiveImagePath<<endl;
			system("pause");
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(32,32), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
		// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), 
		// Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, 
		// double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
		// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
		vector<float> featureVec; //定义一个Vector容器

		hog.compute(img, featureVec, cv::Size(8,8));  //计算HOG特征
		int featureVecSize = featureVec.size();

		for (int j=0; j<featureVecSize; j++)  
		{  		
			CV_MAT_ELEM( *sampleFeaturesMat, float, i, j ) = featureVec[j]; //把特征复制到sampleFeaturesMat的第i行
		
		}  

		sampleLabelMat->data.fl[i] = 1;

		delete [] positiveImageName;
		delete [] positiveImagePath;

	}
	fclose(fileIn);

	cout<<"end of training for positive samples..."<<endl;

	cout<<"*********************************************************"<<endl;
	cout<<"start to train negative samples..."<<endl;

	char* negativeNamelistPath = new char[200]; //文本文档完整文件路径
	strcpy(negativeNamelistPath, negativePath);
	strcat(negativeNamelistPath, "Namelist.txt"); //产生图像文件名文本文档的完整路径名
	// FILE* fileIn;
	if((fileIn=fopen(negativeNamelistPath, "r"))==NULL) //打开文件，读入，返回返回指向FILE对象的指针
	{
		printf("Can not open file.");
		return 0;
	}

	delete [] negativeNamelistPath;
	
	for (int i=0; i<negativeSampleCount; i++)
	{  
		char* negativeImageName = new char[100]; //图像文件名
		fscanf(fileIn,"%s",negativeImageName); //读取文件，遇到whitespace停止
		char* negativeImagePath = new char[200]; //存储完整文件路径
		strcpy(negativeImagePath, negativePath);
		strcat(negativeImagePath,negativeImageName); //将第二个字符串拼接到第一个后面
		printf("%s\n",negativeImagePath); //打印出完整文件路径名
		
		cv::Mat img = cv::imread(negativeImagePath);
		if(img.data == NULL)
		{
			cout<<"negative image sample load error: "<<negativeImagePath<<endl;
			system("pause");
			continue;
		}

		cv::HOGDescriptor hog(cv::Size(32,32), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);  
		vector<float> featureVec; 

		hog.compute(img,featureVec,cv::Size(8,8));//计算HOG特征
		int featureVecSize = featureVec.size();  

		for ( int j=0; j<featureVecSize; j ++)  
		{  
			CV_MAT_ELEM( *sampleFeaturesMat, float, i + positiveSampleCount, j ) = featureVec[ j ];
		}  

		sampleLabelMat->data.fl[ i + positiveSampleCount ] = -1;

		delete [] negativeImageName;
		delete [] negativeImagePath;
	}  
	fclose(fileIn);

	cout<<"end of training for negative samples."<<endl;
	cout<<"********************************************************"<<endl;
	cout<<"start to train for SVM classifier."<<endl;
	
	//设置训练参数
	//CvSVMParams params;  
	//params.svm_type = CvSVM::C_SVC;  
	//params.kernel_type = CvSVM::LINEAR;  
	//params.C = 0.01;   // 默认时C＝1
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON); 
		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
	CvSVMParams params(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
	cout << "训练参数：" << params.svm_type <<", "<< params.kernel_type<< endl;

	//开始训练SVM
	Mysvm svm;
	svm.train( sampleFeaturesMat, sampleLabelMat, 0, 0, params ); //用SVM分类器训练
	// train(const CvMat* trainData, const CvMat* responses, const CvMat* varIdx=0, 
	// const CvMat* sampleIdx=0, CvSVMParams params=CvSVMParams() )
	char* trainSavePath = new char[200];
	strcpy(trainSavePath, classifierSavePath);
	strcat(trainSavePath, "SVM_HOG.xml");
	svm.save(trainSavePath, 0);  //存储训练结果
	cout<< trainSavePath <<"训练结果保存完毕" << endl;

	delete [] trainSavePath;

	cvReleaseMat(&sampleFeaturesMat);
	cvReleaseMat(&sampleLabelMat);

	int supportVectorCount = svm.get_support_vector_count();
	cout<<"support vector size of SVM："<<supportVectorCount<<endl;
	cout<<"************************ end of training for SVM ******************"<<endl;

	CvMat *sv,*alp,*re;//所有样本特征向量 
	sv  = cvCreateMat(supportVectorCount , 324, CV_32FC1);
	alp = cvCreateMat(1 , supportVectorCount, CV_32FC1);
	re  = cvCreateMat(1 , 324, CV_32FC1);
	CvMat *res  = cvCreateMat(1 , 1, CV_32FC1);

	cvSetZero(sv);
	cvSetZero(re);
  
	for(int i=0; i<supportVectorCount; i++)
	{
		memcpy( (float*)(sv->data.fl+i*324), svm.get_support_vector(i), 324*sizeof(float));	//复制特征向量
	}

	double* alphaArr = svm.get_alpha();   
	int alphaCount = svm.get_alpha_count();

	for(int i=0; i<supportVectorCount; i++)
	{
        alp->data.fl[i] = alphaArr[i];
	}
	cvMatMul(alp, sv, re);	//re = alp*sv

	int posCount = 0;
	for (int i=0; i<324; i++)
	{
		re->data.fl[i] *= -1;
	}

	//把re中的数据写到text文档里
	char* SVMDetectorSavePath = new char[200];
	strcpy(SVMDetectorSavePath, classifierSavePath);
	strcat(SVMDetectorSavePath, "SVMDetector.txt");
	FILE* fp = fopen(SVMDetectorSavePath,"wb");
	if( NULL == fp )
	{
		return 1;
	}
	for(int i=0; i<324; i++)
	{
		fprintf(fp,"%f \n",re->data.fl[i]);
	}

	float rho = svm.get_rho();
	fprintf(fp, "%f", rho); //保存分类器
	cout<< SVMDetectorSavePath <<"支持向量保存完毕"<<endl;

	delete [] SVMDetectorSavePath;
	fclose(fp);

	time_t endTime = time(NULL);	//记录结束时间
	cout <<"训练所用时间：" << difftime(endTime, startTime) << "秒" << endl; //打印训练用时

	return 1;
}


Mat& detect(Mat& img, vector<float> SVMDetector)
{	
	time_t startTime = time(NULL);	//记录开始时间

	cv::HOGDescriptor hog(cv::Size(32,32), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	// HOGDescriptor(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), 
	// Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, 
	// bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS)
	hog.setSVMDetector(SVMDetector); //训练的分类器
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//自带的分类器


	cout << "正在检测图像……" << endl;
	vector<cv::Rect>  found; //存储检测结果
	hog.detectMultiScale(img, found, 0.0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
	// detectMultiScale(const GpuMat& img, vector<Rect>& found_locations, double hit_threshold=0, 
	// Size win_stride=Size(), Size padding=Size(), double scale0=1.05, int group_threshold=2)


	if (found.size() > 0) 
	{
		cout << "矩形框数目："<< found.size() << endl;

		//// 保存样本截图
		//for(int i = 0; i < found.size(); i++)
		//{
		//	if((found[i].x >= 0) && ((found[i].x + found[i].width) <= img->width) 
		//		&& (found[i].y >= 0) && (found[i].y + found[i].height <= img->height))
		//	{
		//		char* name = new char[100]; //文件名
		//		char* prePath = new char[200]; // 文件夹路径
		//		itoa(i, name, 10);
		//		strcat(name, ".png"); //构成文件名 i.png
		//		strcpy(prePath, "E:\\SmartCity\\Result\\SaveDetectResult\\"); 
		//		strcat(prePath, name); //形成完整路径

		//		cout << prePath << endl;

		//		/*IplImage * image = cvCreateImageHeader(cvSize(img.cols, img.rows), 8, 3);
		//		image = cvGetImage(&img, image);*/
		//		saveDetectResult(img, found[i], prePath); //保存截图
		//		cout << prePath << " 保存完毕。" << endl;

		//		delete[] name;
		//		delete[] prePath;
		//	}
		//}


		//去除在图像边缘上的方框
		Vector<Rect> found_NoMinus;
		for(int i = 0; i < found.size(); i++)
		{
			if(found[i].x > 0  &&  found[i].y > 0  &&  
				found[i].x + found[i].width < 1920  &&  found[i].y + found[i].height < 1080)
			{
				found_NoMinus.push_back(found[i]);
			}
		}


		//去嵌套
		Vector<Rect> found_NoNest;
		int ii = 0;
		for(int i = 0; i < found_NoMinus.size(); i++)
		{
			Rect r = found_NoMinus[i];
			//下面的这个for语句是找出所有没有嵌套的矩形框r,并放入found_NoNest中,如果有嵌套的
			//话,则取外面最大的那个矩形框放入found_NoNest中
            for(ii = 0; ii <found_NoMinus.size(); ii++)
                if(ii != i && (r&found_NoMinus[ii])==r)
                    break;
            if(ii == found_NoMinus.size())
               found_NoNest.push_back(r);
		}
		cout << "去边界框，去嵌套后，矩形框数目："<< found_NoNest.size() << endl;


		////去重叠
		//Vector<Rect> found_NoOverlap;
		//for (int i = 0; i < found_NoNest.size(); i++)
		//{
		//	MyRect rect1(found_NoNest[i]);
		//	for(int j = 0; j < found_NoNest.size(); j++)
		//	{
		//		MyRect rect2(found_NoNest[j]);
		//		MyRect rect3 = rect1 & rect2;



		//		 

		//	}
		//}
		//cout << "去重叠后，矩形框数目："<< found_NoOverlap.size() << endl;
	


		//在图像上画出矩形
		for(int i = 0; i < found_NoNest.size(); i++)
		{
			rectangle(img, found_NoNest[i].tl(), found_NoNest[i].br(), Scalar(0,255,0), 3);

			//在方框上标出编号
			char* str = new char[100];
			itoa(i, str, 10);
			putText(img, str, found_NoNest[i].br(), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255)); //画出编号
				//void putText(Mat& img, const string& text, Point org, int fontFace, double fontScale, 
				//Scalar color, int thickness = 1, int lineType = 8, bool bottomLeftOrigin = false)
			delete[] str;

			/*Rect r = found[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);*/
		}



	}

	time_t endTime = time(NULL);	//记录结束时间
	cout <<"检测所用时间：" << difftime(endTime, startTime) << "秒" << endl; //打印训练用时

	return img;
}

vector<float> loadSVMDetector(char* path)
{
	vector<float> x;
	ifstream fileIn(path, ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn >> val;
		x.push_back(val);
	}
	fileIn.close();

	if (x.size() == 325)
		cout << path << "分类器导入成功！" << endl;
	else
	{
		cout << "分类器导入错误。" <<endl;
		system("pause");
	}

	return x;
}


