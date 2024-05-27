#include <jwrite/ui/JustWrite.h>
#include <jwrite/ui/ScrollArea.h>
#include <jwrite/ui/BookInfoEdit.h>
#include <jwrite/ui/ColorSchemeDialog.h>
#include <jwrite/ui/MessageBox.h>
#include <jwrite/ColorScheme.h>
#include <jwrite/AppConfig.h>
#include <jwrite/ProfileUtils.h>
#include <widget-kit/TextInputDialog.h>
#include <widget-kit/OverlaySurface.h>
#include <widget-kit/Progress.h>
#include <QScrollBar>
#include <QVBoxLayout>
#include <QKeyEvent>
#include <QDateTime>
#include <QPainter>
#include <QFileDialog>
#include <QCoreApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QApplication>

namespace jwrite::ui {

class BookManager : public InMemoryBookManager {
public:
    bool chapter_cached(int cid) const {
        return chapters_.contains(cid);
    }

    bool is_chapter_dirty(int cid) const {
        return modified_.contains(cid);
    }

    OptionalString fetch_chapter_content(int cid) override {
        if (!has_chapter(cid)) {
            return std::nullopt;
        } else if (chapters_.contains(cid)) {
            return {chapters_.value(cid)};
        } else if (const auto path = get_path_to_chapter(cid); QFile::exists(path)) {
            QFile file(path);
            file.open(QIODevice::ReadOnly | QIODevice::Text);
            Q_ASSERT(file.isOpen());
            const auto content = file.readAll();
            file.close();
            chapters_[cid] = content;
            return content;
        } else {
            return QString{};
        }
    }

    bool sync_chapter_content(int cid, const QString &text) override {
        if (!has_chapter(cid)) { return false; }
        chapters_[cid] = text;
        modified_.insert(cid);
        return true;
    }

    QString get_path_to_chapter(int cid) const {
        QDir dir{AppConfig::get_instance().path(AppConfig::StandardPath::UserData)};
        dir.cd(AbstractBookManager::info_ref().uuid);
        return dir.filePath(QString::number(cid));
    }

private:
    QMap<int, QString> chapters_;
    QSet<int>          modified_;
};

JustWrite::JustWrite() {
    setupUi();
    setupConnections();

    page_map_[PageType::Gallery]->installEventFilter(this);
    page_map_[PageType::Edit]->installEventFilter(this);

    switchToPage(PageType::Gallery);

    //! NOTE: Qt gives out an unexpected minimum height to my widgets and QLayout::invalidate()
    //! could not solve the problem, maybe there is some dirty cached data at the bottom level.
    //! I eventually found that an explicit call to the expcted-to-be-readonly method sizeHint()
    //! could make effects on the confusing issue. so, **DO NOT TOUCH THE CODE** unless you
    //! figure out a better way to solve the problem
    const auto DO_NOT_REMOVE_THIS_STATEMENT = sizeHint();

    requestInitFromLocalStorage();

    command_manager_.load_default();
}

JustWrite::~JustWrite() {
    AppConfig::get_instance().save();

    for (auto book : books_) { delete book; }
    books_.clear();

    jwrite_profiler_dump(QString("jwrite-profiler.%1.log")
                             .arg(QDateTime::currentDateTime().toString("yyyyMMddHHmmss")));
}

void JustWrite::wait(std::function<void()> job) {
    widgetkit::Progress::wait(ui_surface_, job);
}

void JustWrite::updateColorScheme(const ColorScheme &scheme) {
    auto pal = palette();
    pal.setColor(QPalette::Window, scheme.window());
    pal.setColor(QPalette::WindowText, scheme.window_text());
    pal.setColor(QPalette::Base, scheme.text_base());
    pal.setColor(QPalette::Text, scheme.text());
    pal.setColor(QPalette::Highlight, scheme.selected_text());
    pal.setColor(QPalette::Button, scheme.window());
    pal.setColor(QPalette::ButtonText, scheme.window_text());
    setPalette(pal);

    if (auto w = static_cast<QScrollArea *>(page_map_[PageType::Gallery])->verticalScrollBar()) {
        auto pal = w->palette();
        pal.setColor(w->backgroundRole(), scheme.text_base());
        pal.setColor(w->foregroundRole(), scheme.window());
        w->setPalette(pal);
    }

    ui_title_bar_->updateColorScheme(scheme);
    ui_gallery_->updateColorScheme(scheme);
    ui_edit_page_->updateColorScheme(scheme);
}

void JustWrite::updateBookInfo(int index, const BookInfo &info) {
    auto book_info = info;

    if (book_info.author.isEmpty()) {
        book_info.author = getLikelyAuthor();
    } else {
        updateLikelyAuthor(book_info.author);
    }
    if (book_info.title.isEmpty()) { book_info.title = QString("未命名书籍-%1").arg(index + 1); }

    ui_gallery_->updateDisplayCaseItem(index, book_info);

    //! FIXME: only to ensure the sync-to-local-storage is available to access all the book
    //! items, remeber to remove this later as long as realtime sync has been done
    if (const auto &uuid = book_info.uuid; !books_.contains(uuid)) {
        auto bm        = new BookManager;
        bm->info_ref() = book_info;
        books_.insert(uuid, bm);
    } else {
        books_.value(uuid)->info_ref() = book_info;
    }
}

QString JustWrite::getLikelyAuthor() const {
    return likely_author_.isEmpty() ? "佚名" : likely_author_;
}

void JustWrite::updateLikelyAuthor(const QString &author) {
    //! TODO: save likely author to local storage
    if (!author.isEmpty() && likely_author_.isEmpty() && author != getLikelyAuthor()) {
        likely_author_ = author;
    }
}

void JustWrite::toggleMaximize() {
    if (isMinimized()) { return; }
    if (isMaximized()) {
        showNormal();
    } else {
        showMaximized();
    }
}

void JustWrite::setupUi() {
    setObjectName("JustWrite");

    auto top_layout = new QVBoxLayout(this);

    ui_title_bar_  = new TitleBar;
    ui_page_stack_ = new QStackedWidget;
    ui_gallery_    = new Gallery;
    ui_edit_page_  = new EditPage;

    auto gallery_page = new ScrollArea;
    gallery_page->setWidget(ui_gallery_);

    ui_page_stack_->addWidget(gallery_page);
    ui_page_stack_->addWidget(ui_edit_page_);

    top_layout->addWidget(ui_title_bar_);
    top_layout->addWidget(ui_page_stack_);

    ui_surface_        = new widgetkit::OverlaySurface;
    const bool succeed = ui_surface_->setup(ui_page_stack_);
    Q_ASSERT(succeed);

    ui_agent_ = new QWK::WidgetWindowAgent(this);
    ui_agent_->setup(this);
    ui_agent_->setTitleBar(ui_title_bar_);
    ui_agent_->setSystemButton(
        QWK::WidgetWindowAgent::Minimize, ui_title_bar_->systemButton(SystemButton::Minimize));
    ui_agent_->setSystemButton(
        QWK::WidgetWindowAgent::Maximize, ui_title_bar_->systemButton(SystemButton::Maximize));
    ui_agent_->setSystemButton(
        QWK::WidgetWindowAgent::Close, ui_title_bar_->systemButton(SystemButton::Close));

    top_layout->setContentsMargins({});
    top_layout->setSpacing(0);

    page_map_[PageType::Gallery] = gallery_page;
    page_map_[PageType::Edit]    = ui_edit_page_;

    tray_icon_ = new QSystemTrayIcon(this);
    tray_icon_->setIcon(QIcon(":/app.ico"));
    tray_icon_->setToolTip("只写");
    tray_icon_->setVisible(false);

    updateColorScheme(AppConfig::get_instance().scheme());
}

void JustWrite::setupConnections() {
    auto &config = AppConfig::get_instance();

    connect(&config, &AppConfig::on_theme_change, this, [this, &config] {
        updateColorScheme(config.scheme());
    });
    connect(&config, &AppConfig::on_scheme_change, this, &JustWrite::updateColorScheme);
    connect(ui_gallery_, &Gallery::clicked, this, [this](int index) {
        if (index == ui_gallery_->totalItems()) { requestUpdateBookInfo(index); }
    });
    connect(ui_gallery_, &Gallery::menuClicked, this, &JustWrite::requestBookAction);
    connect(this, &JustWrite::pageChanged, this, [this](PageType page) {
        switch (page) {
            case PageType::Gallery: {
                ui_title_bar_->setTitle("只写 丶 阐释你的梦");
            } break;
            case PageType::Edit: {
                const auto &info  = ui_edit_page_->bookSource().info_ref();
                const auto  title = QString("%1\u3000%2 [著]").arg(info.title).arg(info.author);
                ui_title_bar_->setTitle(title);
            } break;
        }
    });
    connect(ui_title_bar_, &TitleBar::minimizeRequested, this, &QWidget::showMinimized);
    connect(ui_title_bar_, &TitleBar::maximizeRequested, this, &JustWrite::toggleMaximize);
    connect(ui_title_bar_, &TitleBar::closeRequested, this, &JustWrite::closePage);
    connect(
        ui_edit_page_, &EditPage::renameTocItemRequested, this, &JustWrite::requestRenameTocItem);
}

void JustWrite::requestUpdateBookInfo(int index) {
    Q_ASSERT(index >= 0 && index <= ui_gallery_->totalItems());
    const bool on_insert = index == ui_gallery_->totalItems();

    auto info = on_insert ? BookInfo{.uuid = AbstractBookManager::alloc_uuid()}
                          : ui_gallery_->bookInfoAt(index);
    if (info.author.isEmpty()) { info.author = getLikelyAuthor(); }
    if (info.title.isEmpty()) { info.title = QString("未命名书籍-%1").arg(index + 1); }

    if (auto opt = BookInfoEdit::getBookInfo(ui_surface_, info)) { updateBookInfo(index, *opt); }
}

void JustWrite::requestBookAction(int index, Gallery::MenuAction action) {
    Q_ASSERT(index >= 0 && index < ui_gallery_->totalItems());
    switch (action) {
        case Gallery::Open: {
            requestStartEditBook(index);
        } break;
        case Gallery::Edit: {
            requestUpdateBookInfo(index);
        } break;
        case Gallery::Delete: {
            const auto choice = MessageBox::show(
                ui_surface_,
                "删除书籍",
                "删除后，作品将无法恢复，请谨慎操作。",
                MessageBox::StandardIcon::Warning);
            if (choice == MessageBox::Yes) {
                const auto uuid = ui_gallery_->bookInfoAt(index).uuid;
                ui_gallery_->removeDisplayCase(index);
                //! NOTE: remove the book-manager means remove the book from the local storage
                //! when the jwrite exits, see syncToLocalStorage()
                //! FIXME: that's not a good idea
                auto bm = books_.value(uuid);
                books_.remove(uuid);
                delete bm;
            }
        } break;
    }
}

void JustWrite::requestStartEditBook(int index) {
    const auto  book_info = ui_gallery_->bookInfoAt(index);
    const auto &uuid      = book_info.uuid;

    if (!books_.contains(uuid)) {
        auto bm        = new BookManager;
        bm->info_ref() = book_info;
        books_.insert(uuid, bm);
    }

    auto bm = books_.value(uuid);
    Q_ASSERT(bm);

    const bool book_changed = ui_edit_page_->resetBookSource(bm);

    switchToPage(PageType::Edit);
    QApplication::processEvents();

    ui_edit_page_->resetWordsCount();

    wait([this, bm] {
        if (const auto &chapters = bm->get_all_chapters(); !chapters.isEmpty()) {
            ui_edit_page_->openChapter(chapters.back());
        }
        ui_edit_page_->flushWordsCount();
    });

    ui_edit_page_->focusOnEditor();
    ui_edit_page_->syncWordsStatus();
}

void JustWrite::requestRenameTocItem(const BookInfo &book_info, int vid, int cid) {
    auto bm = books_.value(book_info.uuid);
    Q_ASSERT(bm);

    const int toc_id = cid == -1 ? vid : cid;
    Q_ASSERT(bm->has_toc_item(toc_id));
    const bool is_volume = cid == -1;

    const auto caption     = is_volume ? "分卷名" : "章节名";
    const auto placeholder = is_volume ? "请输入新分卷名" : "请输入新章节名";
    const auto title       = bm->get_title(toc_id).value();

    const auto opt_new_title =
        widgetkit::TextInputDialog::getInputText(ui_surface_, title, caption, placeholder);

    if (opt_new_title) { ui_edit_page_->renameBookDirItem(toc_id, opt_new_title.value()); }
    ui_edit_page_->focusOnEditor();
}

void JustWrite::requestInitFromLocalStorage() {
    const QDir data_dir{AppConfig::get_instance().path(AppConfig::StandardPath::UserData)};
    if (data_dir.exists()) {
        loadDataFromLocalStorage();
    } else {
        initLocalStorage();
    }
}

void JustWrite::requestQuitApp() {
    hide();

    //! TODO: wait other background jobs to be finished
    syncToLocalStorage();

    QCoreApplication::exit();
}

void JustWrite::initLocalStorage() {
    QDir dir{AppConfig::get_instance().path(AppConfig::StandardPath::UserData)};

    if (!dir.exists()) {
        const bool succeed = dir.mkdir(".");
        Q_ASSERT(succeed);
    }

    QJsonObject local_storage;
    local_storage["major_author"]     = QString{};
    local_storage["last_update_time"] = QDateTime::currentDateTimeUtc().toString();
    local_storage["data"]             = QJsonArray{};

    /*! Json Structure
     *  {
     *      "data": [
     *          {
     *              "book_id": "<book-uuid>",
     *              "name": "<name>",
     *              "author": "<author>",
     *              "cover_url": "<cover-url>",
     *              "last_update_time": "<last-update-time>"
     *          }
     *      ],
     *      "major_author" "<major-author>",
     *      "last_update_time": "<last-update-time>"
     *  }
     **/

    QFile data_file(dir.filePath("mainfest.json"));
    data_file.open(QIODevice::WriteOnly | QIODevice::Text);

    data_file.write(QJsonDocument(local_storage).toJson());

    data_file.close();
}

void JustWrite::loadDataFromLocalStorage() {
    QDir dir{AppConfig::get_instance().path(AppConfig::StandardPath::UserData)};
    Q_ASSERT(dir.exists());

    Q_ASSERT(dir.exists("mainfest.json"));

    QFile data_file(dir.filePath("mainfest.json"));
    data_file.open(QIODevice::ReadOnly | QIODevice::Text);
    const auto text = data_file.readAll();
    auto       json = QJsonDocument::fromJson(text);
    data_file.close();

    Q_ASSERT(json.isObject());
    const auto &local_storage = json.object();

    likely_author_        = local_storage["major_author"].toString("");
    const auto &book_data = local_storage["data"].toArray({});

    for (const auto &ref : book_data) {
        Q_ASSERT(ref.isObject());
        const auto &book = ref.toObject();
        const auto &uuid = book["book_id"].toString("");
        const auto &name = book["name"].toString("");
        Q_ASSERT(!uuid.isEmpty() && !name.isEmpty());

        BookInfo book_info{
            .uuid      = uuid,
            .title     = name,
            .author    = book["author"].toString(""),
            .cover_url = book["cover_url"].toString(""),
        };

        updateBookInfo(ui_gallery_->totalItems(), book_info);

        if (!dir.exists(uuid)) { continue; }

        const bool succeed = dir.cd(uuid);
        Q_ASSERT(succeed);

        auto bm = books_.value(uuid);
        Q_ASSERT(bm);

        if (dir.exists("TOC")) {
            const auto toc_path = dir.filePath("TOC");
            QFile      toc_file(toc_path);
            toc_file.open(QIODevice::ReadOnly | QIODevice::Text);
            const auto toc_text = toc_file.readAll();
            toc_file.close();

            const auto toc_json = QJsonDocument::fromJson(toc_text);
            Q_ASSERT(toc_json.isArray());
            const auto &volumes = toc_json.array();

            /*! Json Structure
             *  [
             *      {
             *          'vid': <volume-id>,
             *          'title': '<volume-title>',
             *          'chapters': [
             *              {
             *                  'cid': <chapter-id>,
             *                  'title': '<chapter-title>',
             *              }
             *          ]
             *      }
             *  ]
             */

            int vol_index = 0;
            for (const auto &vol_ref : volumes) {
                Q_ASSERT(vol_ref.isObject());
                const auto &volume    = vol_ref.toObject();
                const auto  vid       = volume["vid"].toInt();
                const auto  vol_title = volume["title"].toString("");
                bm->add_volume_as(vol_index++, vid, vol_title);
                const auto &chapters   = volume["chapters"].toArray({});
                int         chap_index = 0;
                for (const auto &chap_ref : chapters) {
                    Q_ASSERT(chap_ref.isObject());
                    const auto &chapter    = chap_ref.toObject();
                    const auto  cid        = chapter["cid"].toInt();
                    const auto  chap_title = chapter["title"].toString("");
                    bm->add_chapter_as(vid, chap_index++, cid, chap_title);
                }
            }
        }

        dir.cdUp();
    }
}

void JustWrite::syncToLocalStorage() {
    QJsonObject local_storage;
    QJsonArray  book_data;

    //! FIXME: combine with local data

    for (const auto &[uuid, bm] : books_.asKeyValueRange()) {
        const auto &book_info = bm->info_ref();
        Q_ASSERT(uuid == book_info.uuid);
        QJsonObject book;
        book["book_id"]   = book_info.uuid;
        book["name"]      = book_info.title;
        book["author"]    = book_info.author;
        book["cover_url"] = book_info.cover_url;
        //! FIXME: record real update time
        book["last_update_time"] = QDateTime::currentDateTime().toString("yyyy-MM-dd.HH:mm:ss");
        //! TODO: export toc and contents
        book_data.append(book);
    }

    local_storage["major_author"]     = likely_author_;
    local_storage["last_update_time"] = QDateTime::currentDateTimeUtc().toString();
    local_storage["data"]             = book_data;

    QDir dir{AppConfig::get_instance().path(AppConfig::StandardPath::UserData)};
    Q_ASSERT(dir.exists());

    //! TODO: check validity of local storage file

    QFile data_file(dir.filePath("mainfest.json"));
    data_file.open(QIODevice::WriteOnly | QIODevice::Text);
    data_file.write(QJsonDocument(local_storage).toJson());
    data_file.close();

    //! NOTE: here we simply sync to local according to the book set in the memory, and remove
    //! the book from the set also means remove the book from the local storage, however, in the
    //! current edition, we simply delete the record without removing the content from your
    //! machine, so you can mannually recover it by adding the record to the mainfest.json file
    //! FIXME: you know what I'm gonna say - yeah, that's not a good idea

    for (const auto &[uuid, bm] : books_.asKeyValueRange()) {
        if (!dir.exists(uuid)) { dir.mkdir(uuid); }
        dir.cd(uuid);

        QJsonArray volumes;
        for (const int vid : bm->get_volumes()) {
            QJsonObject volume;
            QJsonArray  chapters;
            for (const int cid : bm->get_chapters_of_volume(vid)) {
                QJsonObject chapter;
                chapter["cid"]   = cid;
                chapter["title"] = bm->get_title(cid).value().get();
                chapters.append(chapter);
            }
            volume["vid"]      = vid;
            volume["title"]    = bm->get_title(vid).value().get();
            volume["chapters"] = chapters;
            volumes.append(volume);
        }

        QFile toc_file(dir.filePath("TOC"));
        toc_file.open(QIODevice::WriteOnly | QIODevice::Text);
        toc_file.write(QJsonDocument(volumes).toJson());
        toc_file.close();

        //! FIXME: unsafe cast
        const auto book_manager = static_cast<BookManager *>(bm);
        for (const int cid : book_manager->get_all_chapters()) {
            if (!book_manager->is_chapter_dirty(cid)) { continue; }
            Q_ASSERT(book_manager->chapter_cached(cid));
            const auto content = std::move(book_manager->fetch_chapter_content(cid).value());
            QFile      file(book_manager->get_path_to_chapter(cid));
            file.open(QIODevice::WriteOnly | QIODevice::Text);
            Q_ASSERT(file.isOpen());
            file.write(content.toUtf8());
            file.close();
        }

        dir.cdUp();
    }
}

void JustWrite::switchToPage(PageType page) {
    Q_ASSERT(page_map_.contains(page));
    Q_ASSERT(page_map_.value(page, nullptr));
    if (auto w = page_map_[page]; w != ui_page_stack_->currentWidget()) {
        ui_page_stack_->setCurrentWidget(w);
    }
    current_page_ = page;
    emit pageChanged(page);
}

void JustWrite::closePage() {
    switch (current_page_) {
        case PageType::Edit: {
            ui_edit_page_->syncAndClearEditor();
            switchToPage(PageType::Gallery);
        } break;
        case PageType::Gallery: {
            //! TODO: throw a dialog to confirm quiting the jwrite
            //! TODO: sync book content to local storage
            requestQuitApp();
        } break;
    }
}

void JustWrite::showEvent(QShowEvent *event) {
    tray_icon_->hide();
}

void JustWrite::hideEvent(QHideEvent *event) {
    tray_icon_->show();
}

bool JustWrite::eventFilter(QObject *watched, QEvent *event) {
    if (event->type() == QEvent::KeyPress) {
        auto e = static_cast<QKeyEvent *>(event);
        if (auto opt = command_manager_.match(e)) {
            if (*opt == GlobalCommand::ShowColorSchemeDialog) {
                auto      &config     = AppConfig::get_instance();
                const auto old_theme  = config.theme();
                const auto old_scheme = config.scheme();

                auto dialog = std::make_unique<ColorSchemeDialog>(old_theme, old_scheme, this);

                connect(
                    dialog.get(),
                    &ColorSchemeDialog::applyRequested,
                    this,
                    [this, &config](ColorTheme theme, const ColorScheme &scheme) {
                        config.set_theme(theme);
                        config.set_scheme(scheme);
                    });

                const int result = dialog->exec();

                if (result != QDialog::Accepted) {
                    config.set_theme(old_theme);
                    config.set_scheme(old_scheme);
                    updateColorScheme(old_scheme);
                } else {
                    config.set_theme(dialog->getTheme());
                    config.set_scheme(dialog->getScheme());
                    updateColorScheme(config.scheme());
                }

                return true;
            }
        }
    }
    return false;
}

} // namespace jwrite::ui
