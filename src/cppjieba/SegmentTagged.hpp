#ifndef CPPJIEBA_SEGMENTTAGGED_H
#define CPPJIEBA_SEGMENTTAGGED_H

#include "SegmentBase.hpp"
#include "DictTrie.hpp"

namespace cppjieba {

class SegmentTagged : public SegmentBase {
public:
    SegmentTagged() {}

    virtual ~SegmentTagged() {}

    virtual bool Tag(const string& src, vector<pair<string, string>>& res) const = 0;

    virtual const DictTrie* GetDictTrie() const = 0;

}; // class SegmentTagged

} // namespace cppjieba

#endif
