
void extract_red_area(const cv::Mat &image, cv::Mat &redMat) {
	int width = image.cols;
	int height = image.rows;
	int x, y;
	double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
	redMat = Mat::zeros(image.size(), CV_8UC1);
	for (x = 0; x < height; x++)
	{
		for (y = 0; y < width; y++)
		{
			B = image.at<Vec3b>(x, y)[0];
			G = image.at<Vec3b>(x, y)[1];
			R = image.at<Vec3b>(x, y)[2];
			RGB2HSV(R, G, B, H, S, V);
			//红色范围，范围参考的网上。可以自己调
			if ((H >= 312 && H <= 360) && (S >= 17 && S <= 100) && (V > 18 && V < 100))
				redMat.at<uchar>(x, y) = 255;
		}
	}


	// 2.提取轮廓
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(
		redMat,               // 输入二值图
		contours,             // 存储轮廓的向量
		hierarchy,            // 轮廓层次信息
		RETR_TREE,            // 检索所有轮廓并重建嵌套轮廓的完整层次结构
		CHAIN_APPROX_NONE);   // 每个轮廓的全部像素
	printf("find %d contours \n", contours.size());

// 过滤轮廓
void filter_contours(std::vector<cv::Vec4i> &hierarchy, std::vector<std::vector<cv::Point>> &contours) {
	std::vector<std::vector<cv::Point>>::iterator itc = contours.begin();
	std::vector<cv::Vec4i>::iterator itc_hierarchy = hierarchy.begin();
	int i = 0;
	int min_size = 20;
	int max_size = 500;
	while (itc_hierarchy != hierarchy.end())
	{
		//验证轮廓大小
		//if (!(hierarchy[i][2] < 0 && hierarchy[i][3] < 0)) // 存在子轮廓/父轮廓
		if (hierarchy[i][3] > 0) // 存在父轮廓
		{
			itc = contours.erase(itc);
			itc_hierarchy = hierarchy.erase(itc_hierarchy);
		}
		else
		{
			//验证轮廓大小
			if (itc->size() < min_size || itc->size() > max_size)
			{
				itc = contours.erase(itc);
				itc_hierarchy = hierarchy.erase(itc_hierarchy);
			}
			else
			{
				++itc;
				++itc_hierarchy;
			}
 
			++i;
		}
	}
}
void draw_center(cv::Mat &dstImg, std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point> &vec) {
	std::vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f rect[4];
	for (int i = 0; i < contours.size(); i++)
	{
		box[i] = minAreaRect(Mat(contours[i])); 
		cv::circle(dstImg, Point(box[i].center.x, box[i].center.y), 2, cv::Scalar(0, 255, 255), cv::FILLED);
		box[i].points(rect);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(255, 0, 255), 1, 8);  //绘制最小外接矩形每条边
		}
		std::cout << "angel:" << box[i].angle << std::endl;
	}
}
