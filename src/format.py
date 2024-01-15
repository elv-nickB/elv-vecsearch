from typing import Any, Dict, List, Optional
from marshmallow import Schema, fields

class SearchArgs(Schema):
    terms = fields.Str()
    filters = fields.Str()
    ids = fields.Method(deserialize='listify')
    max_total = fields.Int()
    search_fields = fields.Method(deserialize='listify')
    auth = fields.Str()

    def listify(self, item: str) -> List[str]:
        return item.split(',') 
    """
    Optional('terms'): str,
    Optional('filters'): str,
    Optional('sort'): str,
    Optional('stats'): str,
    Optional('start'): int,
    Optional('limit'): int,
    Optional('max_total'): int,
    Optional('display_fields'): List[str],
    Optional('search_fields'): List[str],
    Optional('rank_fields'): bool,
    Optional('select'): List[str],
    Optional('remove'): List[str],
    Optional('ids'): List[str],
    """


class SearchOutput(Schema):
    pagination = fields.Dict(fields.Int(), fields.Int())
    results = fields.List(fields.Dict(fields.Str(), fields.Str()))
    stats = fields.Dict(fields.Str(), fields.Dict(fields.Str(), fields.Str()))
    warnings = fields.List(fields.Str())

"""
SearchArgs = Dict[str, Any]
SearchArgsFormat = Schema({
    Optional('terms'): str,
    Optional('filters'): str,
    Optional('sort'): str,
    Optional('stats'): str,
    Optional('start'): int,
    Optional('limit'): int,
    Optional('max_total'): int,
    Optional('display_fields'): List[str],
    Optional('search_fields'): List[str],
    Optional('rank_fields'): bool,
    Optional('select'): List[str],
    Optional('remove'): List[str],
    Optional('ids'): List[str],
}) 

SearchOutput = Dict[str, Any]
SearchOutputFormat = Schema({
    'pagination': {
        'max_total': int,
        'start': int,
        'limit': int,
        'total': int,
    },
    'results': {
        "fields": {
            str, List[str]
        },
        "hash": str,
        "id": str,
        "meta": Dict[str, Any],
        "prefix": str,
        "score": float,
        "qlib_id": str,
        "type": str
    },
    'stats': {
        str: {
            'min': str,
            'max': str,
            'total': int,
            'unique': int,
            'histogram': {
                str: int
            }
        },
    },
    'warnings': List[str],
})
"""