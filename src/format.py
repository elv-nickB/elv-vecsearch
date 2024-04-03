from typing import List
from marshmallow import Schema, fields as field

class Any(field.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        return value

class SearchArgs(Schema):
    terms = field.Str(default='', missing='')
    filters = field.Str()
    ids = field.Method(deserialize='listify')
    max_total = field.Int(missing=10)
    search_fields = field.Method(deserialize='listify')
    authorization = field.Str()
    display_fields = field.Method(deserialize='listify', missing=['all'])
    start = field.Int()
    limit = field.Int()
    num_retrieve = field.Int()
    sort = field.Str()
    stats = field.Str()
    debug = field.Bool(default=False, missing=False)
    uids = field.Method(deserialize='listify', missing=[], default=[])
    clips = field.Bool(default=False, missing=False)
    clips_include_source_tags = field.Bool(default=False, missing=False)
    clips_max_duration = field.Int()

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

class ClipSearchOutput(Schema):
    class Pagination(Schema):
        max_total = field.Int()
        start = field.Int()
        limit = field.Int()
        total = field.Int()
        total_clips = field.Int()
    class Content(Schema):
        fields = field.Dict(Any(), field.List(Any()))
        hash = field.Str()
        id = field.Str()
        meta = field.Dict()
        prefix = field.Str()
        score = field.Float()
        rank = field.Int()
        qlib_id = field.Str()
        type = field.Str()
        url = field.Str()
        image_url = field.Str()
        start = field.Str()
        end = field.Str()
        start_time = field.Int()
        end_time = field.Int()
        source_count = field.Int()
        class Source(Schema):
            prefix = field.Str()
            fields = field.Dict(Any(), field.List(Any()))
        sources = field.List(field.Nested(Source))
    class Stats(Schema):
        min = Any(allow_none=True)
        max = Any(allow_none=True)
        total = field.Int()
        unique = field.Int()
        histogram = field.Dict(Any(), field.Int())
    pagination = field.Nested(Pagination)
    contents = field.List(field.Nested(Content))
    stats = field.Dict(field.Str(), field.Nested(Stats))
    warnings = field.List(field.Str())
    debug = field.Dict(default={})