#include <widget-kit/OverlaySurface.h>
#include <widget-kit/OverlayDialog.h>
#include <widget-kit/LayoutHelper.h>
#include <QStackedLayout>
#include <QPainter>

namespace widgetkit {

OverlaySurface::OverlaySurface()
    : QWidget()
    , overlay_{nullptr}
    , ui_widget_(nullptr)
    , ui_color_{Qt::black}
    , ui_opacity_{100} {
    setupUi();
    setVisible(false);
}

OverlaySurface::~OverlaySurface() {}

bool OverlaySurface::setup(QWidget *widget) {
    if (widget == this || widget == parentWidget() || ui_widget_ || !widget) { return false; }

    auto parent = widget->parentWidget();
    if (!parent) { return false; }

    ui_widget_ = widget;

    auto container = new QWidget;
    auto layout    = new QStackedLayout(container);

    parent->layout()->replaceWidget(ui_widget_, container);

    layout->addWidget(ui_widget_);
    layout->addWidget(this);
    layout->setCurrentWidget(this);
    layout->setStackingMode(QStackedLayout::StackAll);

    hide();

    return true;
}

void OverlaySurface::setColor(const QColor &color) {
    ui_color_ = color;
    update();
}

void OverlaySurface::setOpacity(double opacity) {
    ui_opacity_ = qBound<int>(0, opacity * 255, 255);
    update();
}

void OverlaySurface::reload(OverlayDialog *overlay) {
    if (overlay_) {
        Q_ASSERT(ui_layout_->count() == 3);
        const auto w = ui_layout_->takeAt(1)->widget();
        Q_ASSERT(w == overlay_);
        overlay_->setVisible(false);
        disconnect(overlay_, &OverlayDialog::accepted, this, nullptr);
        disconnect(overlay_, &OverlayDialog::rejected, this, nullptr);
    }

    if (overlay) {
        Q_ASSERT(ui_layout_->count() == 2);
        ui_layout_->insertWidget(1, overlay);
        connect(overlay, &OverlayDialog::accepted, this, &OverlaySurface::closeOverlay);
        connect(overlay, &OverlayDialog::rejected, this, &OverlaySurface::closeOverlay);
    }

    overlay_ = overlay;

    force_update_geometry(this);
}

bool OverlaySurface::showOverlay() {
    if (!overlay_) { return false; }

    show();

    overlay_->updateGeometry();

    overlay_->setVisible(true);
    overlay_->raise();
    overlay_->activateWindow();
    overlay_->setFocus();

    return true;
}

void OverlaySurface::closeOverlay() {
    if (!overlay_) { return; }

    hide();
    reload(nullptr);

    //! TODO: delete overlay is required?
}

void OverlaySurface::setupUi() {
    auto container = new QWidget;
    auto v_layout  = new QVBoxLayout(this);
    auto h_layout  = new QHBoxLayout(container);

    v_layout->addStretch();
    v_layout->addWidget(container);
    v_layout->addStretch();

    h_layout->addStretch();
    h_layout->addStretch();

    v_layout->setContentsMargins({});
    h_layout->setContentsMargins({});

    ui_layout_ = h_layout;

    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
}

void OverlaySurface::paintEvent(QPaintEvent *event) {
    QPainter p(this);

    auto overlay_color = ui_color_;
    overlay_color.setAlpha(ui_opacity_);

    p.fillRect(rect(), overlay_color);
}

void OverlaySurface::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    if (overlay_) { force_update_geometry(this); }
}

} // namespace widgetkit
