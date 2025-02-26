#pragma once

#include <jwrite/TextViewEngine.h>
#include <jwrite/RwLock.h>
#include <QKeyEvent>
#include <QMap>
#include <optional>
#include <stack>

namespace jwrite {

enum class TextInputCommand {
    Reject,
    InsertPrintable,
    InsertTab,
    InsertNewLine,
    InsertBeforeBlock,
    InsertAfterBlock,
    Cancel,
    Undo,
    Redo,
    Copy,
    Cut,
    Paste,
    ScrollUp,
    ScrollDown,
    MoveToPrevChar,
    MoveToNextChar,
    MoveToPrevWord,
    MoveToNextWord,
    MoveToPrevLine,
    MoveToNextLine,
    MoveToStartOfLine,
    MoveToEndOfLine,
    MoveToStartOfBlock,
    MoveToEndOfBlock,
    MoveToStartOfDocument,
    MoveToEndOfDocument,
    MoveToPrevPage,
    MoveToNextPage,
    MoveToPrevBlock,
    MoveToNextBlock,
    DeletePrevChar,
    DeleteNextChar,
    DeletePrevWord,
    DeleteNextWord,
    DeleteToStartOfLine,
    DeleteToEndOfLine,
    DeleteToStartOfBlock,
    DeleteToEndOfBlock,
    DeleteToStartOfDocument,
    DeleteToEndOfDocument,
    SelectPrevChar,
    SelectNextChar,
    SelectPrevWord,
    SelectNextWord,
    SelectToPrevLine,
    SelectToNextLine,
    SelectToStartOfLine,
    SelectToEndOfLine,
    SelectToStartOfBlock,
    SelectToEndOfBlock,
    SelectBlock,
    SelectPrevPage,
    SelectNextPage,
    SelectToStartOfDoc,
    SelectToEndOfDoc,
    SelectAll,
};

class TextInputCommandManager {
public:
    void                        load_default();
    bool                        insert_or_update(QKeySequence key, TextInputCommand cmd);
    bool                        conflicts(QKeySequence key) const;
    std::optional<QKeySequence> keybindings(TextInputCommand cmd) const;
    void                        clear();

    virtual TextInputCommand match(QKeyEvent *e) const;

    static bool  is_printable_char(QKeyCombination e);
    static QChar translate_printable_char(QKeyEvent *e);

private:
    QMap<QKeySequence, TextInputCommand> key_to_cmd_;
    QMap<TextInputCommand, QKeySequence> cmd_to_key_;
};

class GeneralTextInputCommandManager : public TextInputCommandManager {
public:
    TextInputCommand match(QKeyEvent *e) const override;

    void push(TextViewEngine *engine);
    void pop();

    size_t total_saved_context() const;

private:
    mutable core::RwLock         lock_state_;
    std::stack<TextViewEngine *> engine_stack_;
};

}; // namespace jwrite
