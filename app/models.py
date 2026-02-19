# Tortoise ORM database models
from tortoise.models import Model
from tortoise import fields


class Paper(Model):
    id = fields.IntField(primary_key=True)
    title = fields.CharField(max_length=500)
    abstract = fields.TextField()
    authors = fields.JSONField()  # list of strings
    year = fields.IntField()
    created_at = fields.DatetimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    class Meta:
        table = "papers"
