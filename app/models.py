r"""
Defines `Paper` as a Tortoise ORM model. This is hwat maps to the actual `papers` table in SQLite. Each fied (`id`,`title`,`abstract`,`authors`,`year`,`created_at`) becomes a column. Tortoise handles all the SQL.
"""

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


class Chunk(Model):
    id = fields.IntField(primary_key=True)
    paper = fields.ForeignKeyField("models.Paper", related_name="chunks")
    text = fields.TextField()
    index = fields.IntField()  # position of the chunk in the paper (0-based)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "chunks"
