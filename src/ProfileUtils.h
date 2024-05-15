#pragma once

#include <QList>
#include <QTimer>
#include <chrono>
#include <array>
#include <magic_enum.hpp>

#ifndef NDEBUG
#define ON_DEBUG(...) __VA_ARGS__
#else
#define ON_DEBUG(...)
#endif

#define setup_jwrite_profiler()        ON_DEBUG(JwriteProfiler.setup())
#define jwrite_profiler_start(target)  ON_DEBUG(JwriteProfiler.start(ProfileTarget::target))
#define jwrite_profiler_record(target) ON_DEBUG(JwriteProfiler.record(ProfileTarget::target))

#ifndef NDEBUG

namespace jwrite {

enum class ProfileTarget {
    IME2UpdateDelay,
    FrameRenderCost,
    PrepareRenderData,
    TextEngineRenderCost,
    TextBodyRenderCost,
    SelectionAreaRenderCost,
    CursorRenderCost,
    SelectionLocatingCost,
    GetLocationAtPos,
    GeneralTextEdit,
    SwitchChapter,
};

class Profiler : public QObject {
public:
    using timestamp_t     = decltype(std::chrono::system_clock::now());
    using duration_t      = std::chrono::microseconds;
    using duration_list_t = QList<duration_t>;
    using start_record_t  = std::array<timestamp_t, magic_enum::enum_count<ProfileTarget>()>;
    using profile_data_t  = std::array<duration_list_t, magic_enum::enum_count<ProfileTarget>()>;

    void setup();
    void start(ProfileTarget target);
    void record(ProfileTarget target);

protected:
    int   total_valid() const;
    int   indexof(ProfileTarget target) const;
    float averageof(ProfileTarget target) const;
    void  clear(ProfileTarget target);

private:
    start_record_t start_record_;
    profile_data_t profile_data_;
    QTimer        *timer_ = nullptr;
};

} // namespace jwrite

extern jwrite::Profiler JwriteProfiler;

#endif
