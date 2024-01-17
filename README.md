Eluvio vector search API

### Example

##### Build Index 

Vector search indices are based off of on fabric indices. 

1. Make sure index exists on fabric. (QID)
2. Obtain a token for this index (AUTH)
    e.g. `elv content token create <QID>`
3. Call update endpoint `curl "<HOST>/q/<QID>/search_update?auth=<AUTH>"`
4. Check status `curl "<HOST>/q/<QID>/update_status?auth=<AUTH>"`
5. (Stop if needed) `curl "<HOST>/q/<QID>/stop_update?auth=<AUTH>"`

##### Search 

The API is nearly identical to the on-fabric search api. Consult https://hub.doc.eluv.io/content/rep/search_api/ for more information. 

Example:

```
curl "http://127.0.0.1:8085/q/<QID>/search?terms=tom+cruise+playing+poker&search_fields=f_speech_to_text,f_celebrity,f_segment,f_object,f_logo&max_total=10&display_fields=f_speech_to_text,f_celebrity,f_segment&stats=f_display_title_as_string&auth=<AUTH>"
```