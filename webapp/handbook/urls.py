from django.urls import path

from webapp.handbook import views

app_name = "handbook"

urlpatterns = [
    path("", views.home, name="home"),
    path("source/<slug:doc_id>/", views.source, name="source"),
    path("source/<slug:doc_id>/file", views.source_file, name="source_file"),
    path("api/ask", views.api_ask, name="api_ask"),
    path("api/ask/stream", views.api_ask_stream, name="api_ask_stream"),
    path("api/health", views.api_health, name="api_health"),
    path("api/upload", views.api_upload, name="api_upload"),
    path("api/uploads", views.api_uploads, name="api_uploads"),
    path("api/uploads/<slug:doc_id>/delete", views.api_upload_delete, name="api_upload_delete"),
    path("api/conversation/clear", views.api_conversation_clear, name="api_conversation_clear"),
]
