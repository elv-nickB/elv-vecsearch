from typing import List
from marshmallow import Schema, fields as field

class Any(field.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        return value

class SearchArgs(Schema):
    terms = field.Str()
    filters = field.Str()
    ids = field.Method(deserialize='listify')
    max_total = field.Int(missing=10)
    search_fields = field.Method(deserialize='listify')
    auth = field.Str()
    display_fields = field.Method(deserialize='listify', missing=['all'])
    start = field.Int()
    limit = field.Int()
    sort = field.Str()
    stats = field.Str()
    debug = field.Bool(default=False)

    def listify(self, item: str) -> List[str]:
        return item.split(',') 

class SearchOutput(Schema):
    class Pagination(Schema):
        max_total = field.Int()
        start = field.Int()
        limit = field.Int()
        total = field.Int()
    class Result(Schema):
        fields = field.Dict(Any(), field.List(Any()))
        hash = field.Str()
        id = field.Str()
        meta = field.Dict()
        prefix = field.Str()
        score = field.Float()
        rank = field.Int()
        qlib_id = field.Str()
        type = field.Str()
    class Stats(Schema):
        min = Any(allow_none=True)
        max = Any(allow_none=True)
        total = field.Int()
        unique = field.Int()
        histogram = field.Dict(Any(), field.Int())
    pagination = field.Nested(Pagination)
    results = field.List(field.Nested(Result))
    stats = field.Dict(field.Str(), field.Nested(Stats))
    warnings = field.List(field.Str())
    debug = field.Dict(default={})