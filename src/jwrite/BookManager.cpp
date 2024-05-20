#include <jwrite/BookManager.h>
#include <QUuid>

namespace jwrite {

QString AbstractBookManager::alloc_uuid() {
    return QUuid::createUuid().toString();
}

QList<int> InMemoryBookManager::get_all_chapters() const {
    QList<int> results;
    for (auto vid : vid_list_) { results << get_chapters_of_volume(vid); }
    return results;
}

int InMemoryBookManager::add_volume(int index, const QString &title) {
    Q_ASSERT(index >= 0 && index <= vid_list_.size());
    const int id = title_pool_.size();
    title_pool_.insert(id, title);
    vid_list_.insert(index, id);
    cid_list_set_.insert(id, {});
    return id;
}

int InMemoryBookManager::add_chapter(int vid, int index, const QString &title) {
    Q_ASSERT(cid_list_set_.contains(vid));
    auto &cid_list = cid_list_set_[vid];
    Q_ASSERT(index >= 0 && index <= cid_list.size());
    const int id = title_pool_.size();
    title_pool_.insert(id, title);
    cid_list.insert(index, id);
    return id;
}

int InMemoryBookManager::remove_volume(int vid) {
    if (!has_volume(vid)) { return 0; }

    const auto &chaps       = get_chapters_of_volume(vid);
    const int   total_chaps = chaps.size();
    for (const auto cid : chaps) { title_pool_.remove(cid); }
    title_pool_.remove(vid);

    const int index = vid_list_.indexOf(vid);
    vid_list_.remove(index);
    cid_list_set_.remove(vid);

    return total_chaps + 1;
}

bool InMemoryBookManager::remove_chapter(int cid) {
    if (!has_chapter(cid)) { return false; }

    title_pool_.remove(cid);

    for (auto &cid_list : cid_list_set_) {
        const int index = cid_list.indexOf(cid);
        if (index == -1) { continue; }
        cid_list.remove(cid);
        return true;
    }

    Q_UNREACHABLE();
    return false;
}

} // namespace jwrite
