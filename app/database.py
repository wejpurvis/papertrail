# Tortoise configuration to use SQLite


TORTOISE_ORM = {
    "connections": {"default": "sqlite://./papertrail.db"},
    "apps": {
        "models": {
            "models": ["app.models"],
            "default_connection": "default",
        }
    },
}
