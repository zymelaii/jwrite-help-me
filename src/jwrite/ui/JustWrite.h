#pragma once

#include <jwrite/ui/TitleBar.h>
#include <jwrite/ui/EditPage.h>
#include <jwrite/ui/Gallery.h>
#include <jwrite/BookManager.h>
#include <jwrite/GlobalCommand.h>
#include <widget-kit/OverlaySurface.h>
#include <widget-kit/Progress.h>
#include <QWidget>
#include <QStackedWidget>
#include <QStackedLayout>
#include <QSystemTrayIcon>
#include <functional>
#include <QWKWidgets/widgetwindowagent.h>

namespace jwrite::ui {

class JustWrite : public QWidget {
    Q_OBJECT

public:
    enum class TocType {
        Volume,
        Chapter,
    };

    enum class PageType {
        Gallery,
        Edit,
    };

protected:
    void request_create_new_book();
    void do_create_book(BookInfo &book_info);

    void request_remove_book(const QString &book_id);
    void do_remove_book(const QString &book_id);

    void request_open_book(const QString &book_id);
    void do_open_book(const QString &book_id);

    void request_close_opened_book();
    void do_close_book(const QString &book_id);

    void do_update_book_info(BookInfo &book_info);
    void request_update_book_info(const QString &book_id);

    void request_rename_toc_item(const QString &book_id, int toc, TocType type);
    void do_rename_toc_item(const QString &book_id, int toc, const QString &title);

public slots:
    void handle_gallery_on_click(int index);
    void handle_gallery_on_menu_action(int index, Gallery::MenuAction action);
    void handle_gallery_on_load_book(const BookInfo &book_info);
    void handle_book_dir_on_rename_toc_item(const QString &book_id, int toc_id, TocType type);
    void handle_book_dir_on_rename_toc_item__adapter(const BookInfo &book_info, int vid, int cid);
    void handle_on_page_change(PageType page);
    void handle_on_open_settings();

public:
    QString get_default_author() const;
    void    set_default_author(const QString &author, bool force);

    void wait(std::function<void()> job);

    static widgetkit::Progress::Builder waitTaskBuilder() {
        return widgetkit::Progress::Builder{};
    }

public:
    JustWrite();
    ~JustWrite();

signals:
    void pageChanged(PageType page);

public:
    void updateColorScheme(const ColorScheme &scheme);

public:
    void toggleMaximize();

protected:
    void setupUi();
    void setupConnections();
    void requestStartEditBook(int index);

    void requestInitFromLocalStorage();
    void requestQuitApp();

    void initLocalStorage();
    void loadDataFromLocalStorage();
    void syncToLocalStorage();

    void switchToPage(PageType page);
    void closePage();

    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    QMap<PageType, QWidget *>            page_map_;
    PageType                             current_page_;
    QMap<QString, AbstractBookManager *> books_;
    QString                              likely_author_;
    GlobalCommandManager                 command_manager_;
    QSystemTrayIcon                     *tray_icon_;
    TitleBar                            *ui_title_bar_;
    Gallery                             *ui_gallery_;
    EditPage                            *ui_edit_page_;
    widgetkit::OverlaySurface           *ui_surface_;
    QStackedWidget                      *ui_page_stack_;
    QWK::WidgetWindowAgent              *ui_agent_;
};

} // namespace jwrite::ui
