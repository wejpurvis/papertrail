r"""
Defines `Paper` as a Tortoise ORM model. This is hwat maps to the actual `papers` table in SQLite. Each fied (`id`,`title`,`abstract`,`authors`,`year`,`created_at`) becomes a column. Tortoise handles all the SQL.
"""

from tortoise.models import Model
from tortoise import fields


class Paper(Model):
    id = fields.IntField(primary_key=True)
    arxiv_id = fields.CharField(max_length=50, null=True, unique=True)
    title = fields.CharField(max_length=500)
    abstract = fields.TextField()
    authors = fields.JSONField()  # list of strings
    year = fields.IntField()
    created_at = fields.DatetimeField(auto_now_add=True)
    full_text = fields.TextField(null=True)  # for storing extracted text from PDFs

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


class Embedding(Model):
    id = fields.IntField(primary_key=True)
    # OneToOneField means each chunk has exactly one embedding. This is a stricter relationship than ForeignKey - Tortoise will enforce uniqueness
    chunk = fields.OneToOneField("models.Chunk", related_name="embedding")
    vector = fields.JSONField()  # stores embedding as a list of floats
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "embeddings"
