#ifndef TRTS_COMMON_REDUCTION_H
#define TRTS_COMMON_REDUCTION_H

#include <map>

enum ReductionType
{
    SUM,
    MEAN,
    MUL,
    DIV,
    MIN,
    MAX
};

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM},
    {"mean", MEAN},
    {"mul", MUL},
    {"div", DIV},
    {"min", MIN},
    {"max", MAX},
};

const std::map<ReductionType, std::string> REDUCE2reduce = {
    {SUM, "sum"},
    {MEAN, "mean"},
    {MUL, "mul"},
    {DIV, "div"},
    {MIN, "min"},
    {MAX, "max"},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...)          \
    [&] {                                                 \
        switch (reduce2REDUCE.at(reduce))                 \
        {                                                 \
        case SUM:                                         \
        {                                                 \
            static constexpr ReductionType REDUCE = SUM;  \
            return __VA_ARGS__();                         \
        }                                                 \
        case MEAN:                                        \
        {                                                 \
            static constexpr ReductionType REDUCE = MEAN; \
            return __VA_ARGS__();                         \
        }                                                 \
        case MUL:                                         \
        {                                                 \
            static constexpr ReductionType REDUCE = MUL;  \
            return __VA_ARGS__();                         \
        }                                                 \
        case DIV:                                         \
        {                                                 \
            static constexpr ReductionType REDUCE = DIV;  \
            return __VA_ARGS__();                         \
        }                                                 \
        case MIN:                                         \
        {                                                 \
            static constexpr ReductionType REDUCE = MIN;  \
            return __VA_ARGS__();                         \
        }                                                 \
        case MAX:                                         \
        {                                                 \
            static constexpr ReductionType REDUCE = MAX;  \
            return __VA_ARGS__();                         \
        }                                                 \
        }                                                 \
    }()

#endif // TRTS_COMMON_REDUCTION_H