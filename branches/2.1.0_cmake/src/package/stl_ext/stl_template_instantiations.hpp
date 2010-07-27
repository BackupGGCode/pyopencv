#ifndef SD_stl_TEMPLATE_INSTANTIATIONS_HPP
#define SD_SD_stl_TEMPLATE_INSTANTIATIONS_HPP

class stl_dummy_struct {
public:
    struct dummy_struct2 {};
    static const int total_size = 0
        + sizeof(std::vector<int>);
};

#endif
