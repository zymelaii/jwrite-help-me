#include "JustWrite.h"
#include "ProfileUtils.h"
#include <QGuiApplication>
#include <QApplication>
#include <QScreen>
#include <QFontDatabase>

constexpr QSize PREFERRED_CLIENT_SIZE(1000, 600);

QScreen *getCurrentScreen() {
    return QGuiApplication::screenAt(QCursor::pos());
}

QRect getPreferredGeometry(const QRect &parent_geo) {
    const auto w    = qMin(parent_geo.width(), PREFERRED_CLIENT_SIZE.width());
    const auto h    = qMin(parent_geo.height(), PREFERRED_CLIENT_SIZE.height());
    const auto left = parent_geo.left() + (parent_geo.width() - w) / 2;
    const auto top  = parent_geo.top() + (parent_geo.height() - h) / 2;
    return QRect(left, top, w, h);
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    const auto font_name = u8"更纱黑体 SC Light";
    QFontDatabase::addApplicationFont(QString("fonts/%1.ttf").arg(font_name));
    QApplication::setFont(QFont(font_name, 16));

    auto       screen     = getCurrentScreen();
    const auto screen_geo = screen->geometry();

    ON_DEBUG(JwriteProfiler.setup());

    JustWrite client;
    client.setGeometry(getPreferredGeometry(screen_geo));
    client.show();

    return app.exec();
}
