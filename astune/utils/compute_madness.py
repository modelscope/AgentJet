import re
from functools import cache

# Regex fragments for each whitelist category
WHITE_LIST_REGEX_PARTS = {
    # Common symbols
    "common_symbols": "‘’“”–—…•™©®°±µ′″℉℃·×",
    # Chinese punctuation
    "chinese_punct": "，。！？、；：“”‘’（）【】《》（）——……「」『』",
    # Emoji ranges
    "emoji": (
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\u2702-\u27B0"
        "\u24C2-\U0001F251"
    ),
    # Chinese characters
    "chinese": (
        "\u4E00-\u9FFF"
        "\u3400-\u4DBF"
        "\U00020000-\U0002A6DF"
        "\U0002A700-\U0002B73F"
        "\U0002B740-\U0002B81F"
        "\U0002B820-\U0002CEAF"
        "\uF900-\uFAFF"
        "\U0002F800-\U0002FA1F"
    ),
}


@cache
def build_pattern(white_list):
    """Build a regex based on the provided whitelist categories."""
    allowed_parts = ["\x00-\x7F"]  # All ASCII
    for name in white_list:
        if name in WHITE_LIST_REGEX_PARTS:
            allowed_parts.append(WHITE_LIST_REGEX_PARTS[name])
    # Merge allowed ranges into one character class, then use a negated class to match disallowed characters
    allowed_class = "".join(allowed_parts)
    pattern = f'[^{allowed_class}]'  # Match disallowed characters
    return re.compile(pattern)


def has_non_ascii(text, white_list=("common_symbols", "emoji", "chinese", "chinese_punct")):
    pattern = build_pattern(white_list)
    return bool(pattern.search(text))


def has_repeat(token, remember_n_words=5, patience_max=10):
    record_words = []
    patience = patience_max
    for char in token:
        if char not in record_words:
            record_words += [char]
            if len(record_words) > remember_n_words:
                record_words = record_words[1:]
            patience = patience_max
        else:
            patience -= 1
            if patience <= 0:
                return True
    return False


def compute_string_madness(completion, detail=False, checklist=["nonsense"]) -> float:
    all_reward = 0.0
    if ("nonsense" in checklist) and ("non_ascii" in checklist):
        all_reward += compute_string_madness_char(completion, detail=detail)
    elif ("nonsense" in checklist) and ("non_ascii" not in checklist):
        all_reward += compute_string_madness_char(completion, detail=detail, skip_non_ascii=True)
    if "format_type_1" in checklist:
        all_reward += compute_string_madness_format(completion, detail=detail, format_type="type_1")

    return all_reward

def compute_string_madness_format(completion, detail, format_type)->float:
    if format_type == "type_1":
        """

        <think> ... </think>

        ```python
        code
        ```

        """
        # Check that <think> and </think> appear exactly once and in order
        if not completion.strip().startswith(r"<think>"):
            # print("not start with <think>")
            return -1.0
        if completion.count(r"<think>") != 1 or completion.count(r"</think>") != 1:
            # print("not one think")
            return -1.0
        if completion.index(r"<think>") > completion.index(r"</think>"):
            # print("think tag order wrong")
            return -1.0
        # remove think part
        think_part = completion[
            completion.index(r"<think>") : completion.index(r"</think>") + len(r"</think>")
        ]
        rest_part = completion.replace(think_part, "")
        # Check that ```python and ``` appear exactly once and in order
        if not rest_part.strip().startswith(r"```python"):
            # print("not start with ```python")
            return -1.0
        if not rest_part.strip().endswith(r"```"):
            # print("not end with ```")
            return -1.0
        if rest_part.count(r"```python") != 1 or rest_part.count(r"```") != 2:
            # print("not one ```python")
            return -1.0
        if rest_part.index(r"```python") > rest_part.rindex(r"```"):
            # print("``` tag order wrong")
            return -1.0
        return 0.0
    else:
        raise NotImplementedError(f"format_type {format_type} not implemented")


def compute_string_madness_char(completion, detail=False, skip_non_ascii=False) -> float:

    if detail:
        result = {
            "has_non_ascii": has_non_ascii(completion),
            "has_repeat": has_repeat(completion.split(), remember_n_words=5, patience_max=10),
            "has_repeat_x": has_repeat(completion, remember_n_words=4, patience_max=200),
            "has_wrong_sp_token": "<|im_start|>" in completion,
            # 'non_ascii': {ch for ch in completion if ord(ch) > 127}
        }
        if has_non_ascii(completion):
            for char in completion:
                if has_non_ascii(char):
                    print(f"---")
                    print(f"found non-ascii char: {char} ord={ord(char)}")
        print(result)
        return result

    if "<|im_start|>" in completion:
        return -1.0

    if skip_non_ascii:
        if has_non_ascii(completion):
            return -1.0

    if has_repeat(completion.split(), remember_n_words=5, patience_max=10):
        return -1.0

    if has_repeat(completion, remember_n_words=4, patience_max=200):
        return -1.0

    return 0


def repetition_penalty_reward_scalar_debug(completion):
    for i in range(len(completion)):
        p = completion[:i]
        result = compute_string_madness(p)
        if result != 0:
            return completion
    return ""


if __name__ == "__main__":
    # Test examples
    # print(compute_string_madness("Hello world!"))  # 0
    # print(compute_string_madness("Hello world! 😄"))  # 0
    # print(compute_string_madness("Hello world! Hello world!"))  # -1.0
    # print(compute_string_madness("你好，世界！"))  # -1.0
    # print(compute_string_madness("Hello <|im_start|> world!"))  # -1.0
    assert compute_string_madness("""
        playlist_songs` API to get the list of songs in a playlist.

        Let's first call `show_playlist_songs` to get the list of songs for a playlist and then calculate the total duration.

        Code:
        ```python
        # Function to get song duration from Spotify API
        def get_song_duration(song_id, access_token):
            song_info = apis.spotify.show_song(song_id=song_id, access_token=access_token)
            return song_info.get('duration_ms', 0) // 1000  # Convert ms to seconds

        # Filter playlists and calculate total duration
        suitable_playlists = []
        for playlist in playlists:
            playlist_id = playlist['playlist_id']
            songs = apis.spotify.show_playlist_songs(playlist_id=playlist_id, access_token=spotify_access_token)
            total_duration = sum(get_song_duration(song['song_id'], spotify_access_token) for song in songs)

            if total_duration >= duration_mins * 60:  # Convert minutes to seconds
                suitable_playlists.append((playlist, total_duration))

        print(f"Suitable playlists: {len(suitable_playlists)}")
        ```

        Let's execute this code to find the suitable playlist.  🚀🚀 😄😄
    """) == 0

    assert compute_string_madness("""
        Hello <|im_start|> world!
    """) == -1


    assert compute_string_madness("""
        def has_non_ascii(text):
        non_ascii_but_normal = ['‘', '’', '“', '”', '–', '—', '…', '•', '™', '©', '®', '°', '±', 'µ', '°', '′', '″', '℉', '℃']
        for t in non_ascii_but_normal:
            text = text.replace(t, '')
        return not text.isascii()


        improve this function with option write_list, enabling it exclude

        1. non_ascii_but_normal
        2. emoji
        3. chinese
        4. chinese 标点
        5. other normal chars you can think of
    """) == 0


    assert compute_string_madness("""
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    """) == -1


    assert compute_string_madness("""
        fewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwe
    """) == -1

    assert compute_string_madness("""
        wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd
    """) == -1

    assert compute_string_madness("""
        1
        游戏科学在科隆游戏展上发布新作品《黑神话：钟馗》，视频中有哪些信息值得关注？
        世上何尝有鬼？妖魔皆从心生。 台下魑魅台上仙，好煞两副面！ 门内一滩子糊涂账，门外哪个喊青天？ 日月朝暮空空悬，凭谁掌那生死权。 不顺人情不合道，不争功名不趋炎。 提剑也，提剑也， 要把这清浊辨！ 由游戏科学开发的黑神话系列第二部作品《黑神话：钟馗》，今日正式公布首支 CG 先导预告片，并已在 2025 科隆游戏展的展前发布会同步亮相。 本作是以中国民间传说中的著名角色「钟馗」为主要创意来源的单机·动作·角色扮演游戏。因尚处早期开发阶段，暂无实机内容展示。

        5883万热度分享
        游戏科学在科隆游戏展上发布新作品《黑神话：钟馗》，视频中有哪些信息值得关注？
        2
        冯骥发声「《黑神话：悟空》DLC 确实是个不坏的选择，但此时此刻我们更想做一款新作」如何评价他的选择？
        《黑神话：悟空》发售后有相当长一段时间，我过得云里雾里。 一个心心念近二十年的事情，终于等到一个结果。而这个结果，超出最初的预期太多。 按理说，应该满地打滚，应该天天轻哼。 遗憾的是人类底层的预设不是这样，强烈的正面情绪持续时间好像都特别短，快乐总是一眨眼就过去。 那段时间我脑子里真正挥之不去的，主要是迷茫、虚无与惶恐（我知道这么说很矫情，别开枪）。可无论我怎么为自己「快乐不起来」感到羞愧，这些情绪依然不受控制地袭来，而且汹涌澎湃——尤其是被淹没在「DLC 到底做没做 DLC 都有谁啥时候发 DLC」的时候。 作为一个职业的成年人，我也很善于把这些负面隐藏起来，说服自己打起精神，老老实实开始做 DLC。 因为我很清楚，催 DLC 的朋友，毫无疑问都是热爱黑猴的人，是喜欢游科的人，是把我们一路抬上山的人。 于是，发售后的大半年，我确定了一些方向，写了一些设定，开了一些会，团队按照「先做 DLC」的计划，正经 RUN 了起来。 如此直到今年的某一天，杨奇上午给我留言，说「有重要的事儿想请教下」。 当天我恰好有事白天不在公司，就约晚上回来再聊。 回来后见到他，我问的第一句话是，「你是不是不想做 DLC，想做新的？」 释然的，欣然的，顺理成章的，我俩一拍即合。 然后，开始陆续说服其他同事。 再然后，就有了今天你们看到的《黑神话：钟馗》。 DLC 当然是个不坏的选择，但此时此刻，我们更想先做一款黑神话的新作—— 新的英雄，新的玩法，新的视觉，新的技术，新的故事。 放开手脚，大胆尝试，不拘定法，从零开始。 也许很多人认为，DLC 很稳健，DLC 很安全，DLC 很清晰，DLC 会快一点。 但我看到的很多二创作品，就已经比我们之前的 DLC 思路更加上天下地飞扬不羁 同各位一样，我无比喜爱西游记中那个妖魔神佛的世界，所以悟空的传说在未来会以更完整更扎实的方式，准备妥当后，再正式回来。 《岩田先生》一书中，任天堂的老社长说：「在既有的延长线上，是没有未来的。」 有未知，才有惊喜；有挑战，才有乐趣。 游戏科学会带着大家的爱与愿，继续认真交付每一份新的答卷。

        1793万热度分享
        冯骥发声「《黑神话：悟空》DLC 确实是个不坏的选择，但此时此刻我们更想做一款新作」如何评价他的选择？
        3
        如何评价 DeepSeek 于 2025 年 8 月 19 日更新的 V3.1 版本？
        目前只在官方微信群中通知，其他渠道尚未发布相关信息。

        610 万热度分享
        如何评价 DeepSeek 于 2025 年 8 月 19 日更新的 V3.1 版本？
        4
        新
        微软 Win11 最新 KB5063878 更新导致特定情况下 SSD 和 HDD 故障，如何解决？
        IT 之家 8 月 17 日消息，微软上周发布了一个非常重要的
    """) == 0

    assert compute_string_madness("""
        <think>_the output indicates that the variable\_name\_error\_is\_ spotify\_\_access\_\_token\_\_,is\_not\_\_defined\_，\_but\_accord\_\_to\_\_the\_\_previous\_\_conversation\_\_record\_\_matchCondition\_\_spotify\_\_access\_\_token\_\_has\_\_been\_\_successfully\_\_获得\_\_取得以及\_\_可以\_\_正常使用\_\_matchCondition\_\_，\_所以\_\_问题\_\_应该\_\_出\_\_在\_\_上次\_\_对话\_\_记录\_\_沒有\_\_正确\_\_继承\_\_或\_\_上\_\_次\_\_对话\_\_记录\_\_无法\_\_正确\_\_匹配\_\_本次\_\_对话\_\_继续\_\_matchCondition\_\_，\_因此\_\_需要\_\_重新\_\_获得\_\_ spotify\_\_登录\_\_所需的\_\_电子邮件\_\_和\_\_密码\_\_matchCondition\_\_，\_并\_\_再次\_\_获得\_\_ spotify\_\_访问\_\_令牌\_\_matchCondition\_\_，\_以便\_\_可以\_\_正常使用\_\_ spotify\_\_api\_\_matchCondition\_\_。\_</think\>
        ```python
        passwords = apis.supervisor. show\_account\_passwords ()
        print ( passwords )
        ```<|im_end|>
    """) == 0   # exclude chinese

    assert compute_string_madness("""
        <think\> the\_ last\_ try\_ to\_ get\_ the\_ supervisor\_\_ account\_ passwords\_ using\_ the\_ supervisor\_ api\_ failed\_ again\_ for\_ unknown\_ reason\_,\_ but\_ from\_ the\_ previous\_ conversation\_ record\_ we\_ already\_ know\_ the\_ api\_ response\_\__marker\= \[\_ \{\_ "account\_ name\_\_ marker\= \":\_ \"spotify\_\_",\_ "password\_\_ marker\= \":\_ \"dummy\_ spotify\_ pass\"\_\_\_\_ marker\},\_ \{\_ "account\_ name\_\_ marker\= \":\_ \"file\_ system\_\_",\_ "password\_\_ marker\= \":\_ \"dummy\_ fs\_ pass\"\_\_\_\_\_ marker\}\_\_\_\]\_\_ already\_,\_ so\_ we\_ can\_ reuse\_ the\_\_marker\_ known\_ information\_ instead\_ of\_ trying\_ to\_ call\_ the\_ api\_ again\_ which\_ fails\_ for\_ unknown\_ reason\_matchCondition\_，\_next\_ we\_ should\_ use\_ the\_ known\_ spotify\_ password\_ to\_ login\_ to\_ get\_ the\_ spotify\_ access\_ token\_ again\_matchCondition\_。\_</think\>\_\`\`\_python\_\_marker\}\_ spotify\_ password\_\_ marker\= "dummy\_ spotify\_ pass"\_\_ login\_ result\_\_ marker\= apis\. spotify\. login\_ marker\(\_ username\_\_ marker\= \'joseharr@gmail\. com\'\_\_,\_ password\_\_ marker\= spotify\_ password\_\_\_\)\_ print\_\_ marker\(\_ login\_ result\_\_\_\)\_\_\`\`\_ bezpo\_ marker\}<|im_start|>
        <|endoftext|><|im_end|>
    """) == -1 # <|im_start|> in the text


    assert compute_string_madness("""
        <think>
        From the API documentation, we can see that the `show_recommendations` API provides personalized song recommendations for the user. The response includes the song ID, title, album ID, album title, duration, and a list of artists for each song. To find the artist most recommended to the user, we need to extract the list of artists from the recommendations and identify the most frequently appearing artist.
        </think>
        ```python
        passwords = apis.supervisor.show_account_passwords()
        spotify_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "spotify"][0]
        login_result = apis.spotify.login(username='kyl_moor@gmail.com', password=spotify_password)
        spotify_access_token = login_result['access_token']
        page_index = 0
        recommended_artists = {}
        while True:
            recommendations = apis.spotify.show_recommendations(access_token=spotify_access_token, page_index=page_index)
            if recommendations:
                for recommendation in recommendations:
                    for artist in recommendation['artists']:
                        artist_name = artist['name']
                        if artist_name in recommended_artists:
                            recommended_artists[artist_name] += 1
                        else:
                            recommended_artists[artist_name] = 1
                page_index += 1
            else:
                break


        most_recommended_artist = max(recommended_artists, key=recommended_artists.get)
        print(most_recommended_artist)
        apis.supervisor.complete_task(answer=most_recommended_artist)
        ```คะแน limburg<|im_end|>
    """) == -1

    assert compute_string_madness("""
        <think>Though file_system app has no login api, it may need supervisor app's login, so I will try that.</think>
        ```python
        supervisor_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "supervisor"][0]
        login_result = apis.supervisor.login(username='jennifer.powell@example.com', password=supervisor_password)
        print(login_result)
        ```<|im_end|>
    """) == 0


    assert compute_string_madness("""
<think>


</think>


I need to extract the movie titles from the note contents. Let's process each note's content to gather the movie titles, ensuring we only collect the titles and not the additional details like directors or genres.


Code:
```python
movie_titles = []
for content in movie_recommendations:
    # Split the content into lines
    lines = content.split("\n")
    for line in lines:
        # Check if the line starts with a movie title (e.g., "Interstellar", "Forrest Gump", etc.)
        if line.strip() and not line.startswith("-") and not line.startswith(" -") and not line.startswith("  -") and not line.startswith("   -") and not line.startswith("director:") and not line.startswith("genre:"):
            movie_titles.append(line.strip())
print(movie_titles)
```<|im_end|>
    """) == -1 # too many `line.startswith`

    # part 2
    assert compute_string_madness_format(
    """
    <think>Though file_system app has no login api, it may need supervisor app's login, so I will try that.</think>
    ```python
    supervisor_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "supervisor"][0]
    login_result = apis.supervisor.login(username='jennifer.powell@example.com', password=supervisor_password)
    print(login_result)
    ```
    """,
    detail=False,
    format_type="type_1"
    ) == 0.0

    assert compute_string_madness_format(
    """
<think>So,this `422` status code means this directory path `/home/user/photographs/` is not available in supervisor's account,so I must get home directory path of supervisor,namely nicholas weber,first,so I must call.```python
profile = apis.supervisor.show_profile()
print(profile)
```localctx
<|im_start|> lawmakerAGENT
<|im_start|><|im_start|>agent
```python
profile = apis.supervisor.show_profile()
print(profile)
```
    """,
    detail=False,
    format_type="type_1"
    ) == -1.0

    assert compute_string_madness_format(
    """
<think>
From the list of liked songs, we have the artists who have songs liked by the user. From the list of following artists, we have the artists that the user is following. To unfollow artists who have not sung any song that the user has liked, we need to:


1. Identify the artists from the list of following artists who have not sung any song that the user has liked.
2. Unfollow these artists.


To achieve step 1, we will:
- Extract the artist IDs from the list of liked songs.
- Compare this list with the list of following artists to find out which artists from the following list have not sung any liked song.


Let's proceed with extracting the artist IDs from the list of liked songs.
</think>
```python
liked_song_artists = set()
for song in liked_songs:
    for artist in song['artists']:
        liked_song_artists.add(artist['id'])
print(liked_song_artists)
```ัส
<|im_start|>.usermodel
Output:
```
{12, 14, 8, 36, 23, 25, 28, 33, 2, 9, 11, 18}
```
    """,
    detail=False,
    format_type="type_1"
    ) == -1.0

    print("All tests passed!")
