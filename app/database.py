"""
Defines the Tortoise ORM configuration for the app. This is imported and used in `main.py` to initialize the database connection and generate schemas on startup.
"""

TORTOISE_ORM = {
    "connections": {"default": "sqlite://./papertrail.db"},
    "apps": {
        "models": {
            "models": ["app.models"],
            "default_connection": "default",
        }
    },
}
