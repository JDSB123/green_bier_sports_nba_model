"""
Route modules for the NBA prediction API.

This package contains modular route definitions that are incrementally
being extracted from the monolithic app.py file.

Available routers:
- health_router: Health, metrics, and verification endpoints
- admin_router: Admin monitoring and cache management
- meta_router: API metadata, registry, and markets
- tracking_router: Pick tracking and ROI summary

Usage in app.py:
    from src.serving.routes.health import router as health_router
    from src.serving.routes.admin import router as admin_router
    from src.serving.routes.admin import meta_router
    from src.serving.routes.tracking import router as tracking_router

    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(meta_router)
    app.include_router(tracking_router)
"""
