#ifndef SD_cv_h_TEMPLATE_INSTANTIATIONS_HPP
#define SD_cv_h_TEMPLATE_INSTANTIATIONS_HPP

class cv_h_dummy_struct {
public:
    struct dummy_struct2 {};
    static const int total_size = 0
        + sizeof(cv::Point_<float>)
        + sizeof(cv::Point_<int>)
        + sizeof(cv::Rect_<int>)
        + sizeof(cv::Seq<CvConnectedComp>)
        + sizeof(cv::Seq<CvSURFPoint>)
        + sizeof(cv::Size_<int>)
        + sizeof(std::vector<CvConnectedComp>)
        + sizeof(std::vector<CvSURFPoint>);
};

#endif
