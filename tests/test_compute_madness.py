# flake8: noqa

from ajet.utils.compute_madness import (
    compute_string_madness,
    compute_string_madness_format,
)


def test_compute_string_madness_examples():



    assert (
        compute_string_madness(
            r"""
            <think>
            Okay, let's try to solve this problem step by step. So, there are 42 participants sitting in a circle, including Dean. Dean starts with three calculators showing 1, 0, and -1. Each person, including Dean, performs specific operations on these calculators and passes them along. After everyone has had a turn, Dean adds the numbers on the calculators. We need to find that final sum.
            First, let me understand what operations each person does. The problem says that Dean starts by pressing the cube button on the calculator showing 1, square on the one with 0, and negation on the -1 calculator. Then he passes them. Each subsequent person does the same operations: cube the first calculator, square the second, negate the third.
            So, each time a person gets the calculators, they perform the same three operations on the same calculators. Let me note down what each operation does mathematically.
            For the first calculator (initially 1): pressing the cube button would take the current value and cube it. For example, if the value is x, it becomes x³.
            Second calculator (initially 0): pressing the square button would square it, so x².
            Third calculator (initially -1): pressing the negation button would multiply it by -1, so -x.
            But wait, each person does these operations in sequence. So for each of the three calculators, the operations are applied once per person's turn. However, the problem says that each person presses the same buttons on the same calculators. So, each time a person takes the calculators, they perform these three operations in order, right?
            Wait, but the way the problem is phrased: "pressing the cube button on the calculator that shows 1, pressing the square button on the one that shows 0, and on the calculator that shows -1, he presses the negation button." So, each time a person takes the calculators, they perform these three operations on the three calculators. But the key here is that the operations are applied to the current values of the calculators, not the original values.
            So each of the 42 participants (including Dean) will perform these three operations once on the calculators. But the order of passing is important. Let me think about the flow.
            Initially, Dean has the calculators. He does the operations, then passes them to the next person. Then that person does the same operations, and so on, until all 42 people have done their operations, and the calculators come back to Dean. Then he adds them up.
            So, the question is: what is the final value of each calculator after 42 operations (each operation being the cube, square, and negation steps), and then sum them.
            Wait, but each person performs three operations: cube the first, square the second, negate the third. But each of these operations is applied once per person. Therefore, each calculator goes through 42 transformations, but depending on which calculator it is.
            Wait, no. Wait, the three calculators are passed around. Let me clarify:
            When Dean starts, he has three calculators. Each person, when they get the calculators, performs operations on all three calculators. So, each calculator is handled by each person once, but the operations per calculator depend on their position.
            Wait, no. Wait, each person handles all three calculators. So, each calculator is being operated on by each person once, but the operations are different for each calculator.
            Wait, no. For example, the first calculator is the one that initially shows 1. But when Dean passes them to the next person, that person will perform the same operations: cube the first calculator (which now might have a different value), square the second, and negate the third. So each of the three calculators is being operated on by each person once, but each person applies their own set of operations to each calculator. Wait, no, perhaps the same three calculators are being operated on by each person. So, each person, when they receive the calculators, performs the cube on the first calculator, square on the second, and negate on the third. Therefore, each of the three calculators undergoes the same sequence of operations each time they are passed to a person. Wait, no, the operations are applied every time a person takes the calculators.
            So, the first calculator (call it C1) is initially 1. Each person, when they get the calculators, will perform cube on C1. So, each of the 42 people applies cube to C1 once. Similarly, each person applies square to C2 (second calculator) once. And each person applies negation to C3 (third calculator) once.
            Wait, that can't be right. Because each person would be taking the calculators, performing the operations on each of them, and passing them on. So, each calculator is being operated on by each person once. So, for example, C1 starts with 1. Then Dean cubes it, then person 2 cubes it again, and so on, 42 times. Similarly for the other calculators. So each calculator undergoes 42 operations, each of which is cube, square, or negation, depending on which calculator it is.
            Wait, no. Because each person applies the same three operations: cube on the first calculator, square on the second, and negation on the third. So, each time a calculator is passed to a new person, the same operations are applied again. So, for example, C1 is subjected to cube operations every time it's passed to a person. Since there are 42 people, C1 will be cubed 42 times. Similarly, C2 will be squared 42 times, and C3 will be negated 42 times.
            But that would be the case only if each person applies the same operation to each calculator. So the first calculator is always cubed by each person, second is squared, third is negated. Therefore, after 42 people, the first calculator has been cubed 42 times, the second squared 42 times, the third negated 42 times.
            But wait, that seems to be the case. Because each time a calculator is passed to a person, they apply the same operations as Dean did. So, for example, C1 starts as 1. Then Dean cubes it (so becomes 1^3 = 1). Then the next person cubes it again (1^3 = 1), and so on. So after 42 cubings, it's still 1. Similarly, C2 starts at 0. Each person squares it. So 0 squared is 0, so after 42 squares, it's still 0. C3 starts at -1. Each person negates it, so the value alternates between -1 and 1 with each negation. Since there are 42 negations, which is even, the final value would be 1 (because starting at -1, after even number of negations, it's back to -1? Wait wait. Let's think.
            Wait, starting value of C3 is -1. Then each person negates it. So first negation: -(-1) = 1. Second negation: -1. Third: 1. Etc. So after each negation, it alternates. Since there are 42 people, including Dean. So the first negation is done by Dean, then the second by the next person, etc., up to 42 times. So after 42 negations, the value would be (-1)^{42}? Wait no. Because each negation is multiplying by -1. So the value is multiplied by (-1)^{number of negations}. The initial value is -1. So after 42 negations, the value is (-1) * (-1)^{42} = (-1)^{43} = -1. Wait, that's because each negation is multiplying by -1. So total effect is (-1)^{42} multiplied by the initial value. Initial value is -1, so total is (-1) * (-1)^{42} = (-1)^{43} = -1. But 42 is even, so (-1)^{42} is 1, so total is -1 * 1 = -1? Wait no. Wait, let me think again. The initial value is -1. Each negation is multiplying by -1. So after one negation, it's (-1) * (-1) = 1. After two negations: 1 * (-1) = -1. So for 42 negations, which is even, the result is -1. Because after even number of negations, you end up with the original value. Wait, because each two negations cancel each other. So yes, 42 is even, so it would return to -1. But wait, initial value is -1. After 42 negations (even), the value would be -1. So that's what C3 would be.
            Wait, but if that's the case, then the sum would be 1 (from C1) + 0 (from C2) + (-1) (from C3) = 0. But that seems too straightforward, and maybe I made a mistake here. Because that would mean that each calculator is operated on 42 times. But let me check again.
            Wait, the problem says that Dean starts with the calculators. Then he presses the buttons, then passes them to the next person. Then each person does the same. So the first person (Dean) does the operations, then the second person (next in circle) does the same, and so on until all 42 people have done their operations. So each calculator is being operated on once per person. Therefore, each calculator undergoes 42 operations (each time, the same operation: for C1, cube; C2, square; C3, negation). Therefore, for C1, which starts at 1, being cubed 42 times. Let's see.
            Cubing 1 any number of times: 1^3 is 1, 1^3 is 1, etc. So after any number of cubings, C1 remains 1. Similarly, for C2, starting at 0. Squaring 0 any number of times is still 0. So C2 remains 0. For C3, starting at -1, negated 42 times. Since negation is multiplying by -1 each time. So after even number of negations, it's back to original. So since 42 is even, C3 ends up at -1. Therefore, the sum is 1 + 0 + (-1) = 0.
            But that seems too simple. However, the problem states that Dean is included in the 42 participants, and he starts by pressing the buttons, then passes them. So the first operation is Dean's, then the next person, etc., all 42 people. So yes, each calculator is processed 42 times.
            Wait but maybe I have a misunderstanding here. Let me recheck the problem statement.
            "Dean starts by pressing the cube button on the calculator that shows 1, pressing the square button on the one that shows 0, and on the calculator that shows -1, he presses the negation button. After this, he passes all of the calculators to the next person in the circle. Each person presses the same buttons on the same calculators that Dean pressed and then passes them to the next person."
            Ah! So each person presses the same buttons on the same calculators. So, for example, when a person receives the calculators, they press the cube button on the calculator that currently displays 1 (but that calculator might have a different value now)? Or do they press the cube button on the calculator that was originally the one that showed 1?
            Wait, the wording is ambiguous here. The problem says: "pressing the cube button on the calculator that shows 1, pressing the square button on the one that shows 0, and on the calculator that shows -1, he presses the negation button."
            But when the calculators are passed around, the numbers displayed on them change. So, when each person receives them, they look for which calculator shows 1, which shows 0, and which shows -1. But after the first operation, the values may have changed. Wait, but that can't be right, because after Dean does his operations, the calculators' values are:
            Original values:
            C1: 1 → cube → 1³ = 1
            C2: 0 → square → 0² = 0
            C3: -1 → negation → 1
            So after Dean's first operation, the calculators show 1, 0, and 1. Then when the next person comes, they are supposed to press the cube button on the one that shows 1 (but there are two calculators showing 1 now: the original C1 and the original C3). But the problem statement says "the calculator that shows 1, the one that shows 0, and the calculator that shows -1". Wait, but after Dean's first operation, the calculator that showed -1 is now 1, so there's no calculator that shows -1. So this suggests that my initial understanding is incorrect. Therefore, my previous reasoning was wrong.
            Oh no, so the key here is that each person presses the buttons based on the current values of the calculators. That is, each time a person takes the calculators, they look for which calculator shows 1, which shows 0, and which shows -1. But this is only possible if the calculators are being labeled based on their current values. However, after the first operation, the third calculator (originally -1) becomes 1, so there are two calculators showing 1. Then the next person would not know which one to press the cube button on, and which one to press the square button on, etc. Therefore, this suggests that my initial understanding was wrong.
            So, maybe the problem is that the operations are applied to the calculators in a fixed order, regardless of their current values. So, each person always presses the cube button on the first calculator (the one that was initially 1), the square button on the second calculator (initially 0), and the negation button on the third calculator (initially -1). That is, the operations are applied to the same calculator each time, not based on their current value. This would make sense, because otherwise the problem is impossible as the labels change.
            Therefore, the correct interpretation is that each person applies the operations to the same calculators (i.e., the three calculators Dean started with), not based on their current values. So, the first calculator is always the one that was initially 1, the second calculator is always the one that was initially 0, and the third is always the initially -1 calculator. Thus, regardless of their current values, each person applies the same operations to these same calculators.
            This would make the problem feasible. So with this understanding, the operations per calculator are as follows:
            - The first calculator (originally 1) is cubed by each person.
            - The second calculator (originally 0) is squared by each person.
            - The third calculator (originally -1) is negated by each person.
            Therefore, each calculator is subjected to 42 operations (since there are 42 people, including Dean), and each operation is applied once per person.
            Now, let's re-analyze each calculator.
            First calculator: starts at 1. Each person (42 times) cubes it. So we need to compute 1 cubed 42 times. Since 1^3 is always 1, the value remains 1.
            Second calculator: starts at 0. Each person squares it. 0 squared is still 0, so after 42 squarings, it's still 0.
            Third calculator: starts at -1. Each person negates it (multiplies by -1). So after each operation, it flips sign. Since there are 42 operations, which is even, the final value is (-1) * (-1)^{42} ?
            Wait, initial value is -1. Each negation is multiplying by -1. So after 1 negation: (-1)*(-1) = 1. After 2 negations: 1 * (-1) = -1. After 3: -1 * (-1) = 1. So even number of negations brings back to original value. Since 42 is even, the result is -1.
            So sum is 1 + 0 + (-1) = 0. So the final answer is 0?
            But that seems too straightforward, and maybe there's something wrong with my reasoning here.
            Wait, but why is that? Let me think again. The third calculator is being negated 42 times. Starting at -1.
            After each negation, the value alternates between -1 and 1. So if you negate it once: 1. Twice: -1. Three times: 1. Etc. Since there are 42 negations, which is even, the value after 42 negations is -1. Correct.
            So total sum is 1 + 0 + (-1) = 0. But then why is this a competition problem? It seems too easy. Did I miss something?
            Wait, maybe I misunderstood the number of times each operation is applied. Let's think about the passing of the calculators. Dean starts with them, does his operations, then passes them. Then each of the other 41 people does the same operations, passing them on. So total number of times each operation is applied is 42 times (Dean + 41 others). So that's correct.
            Alternatively, maybe the operations are applied once per person, but the passing of the calculators is such that each calculator is passed around the circle once, so each person only touches each calculator once. Wait, but the problem says that Dean passes them to the next person, and each person does the same operations and passes them. So the calculators are passed from Dean to person 2, then to person 3, etc., until they come back to Dean. So each calculator goes through all 42 people. Therefore, each calculator is operated on 42 times. So the previous reasoning holds.
            Alternatively, maybe each person only operates once on the calculators, and the calculators are passed around, but each operation affects the calculator's value. So the first calculator is operated on by each person once, but the operations are applied in sequence.
            But with that, the result for the first calculator is 1 cubed 42 times. Which is 1^3^3^...^3 (42 times). But since exponentiation is right-associative, but in this case, cubing is multiplying the value by itself three times. Wait, no. The operation of pressing the cube button would take the current value and replace it with its cube. So, for example, if the calculator shows x, after pressing the<|im_end|>
            """, detail=True
        )) == 0


    assert (
        compute_string_madness(
            r"""
            <think>
            Okay, let's try to solve this problem. The equation given is 2 sinx siny + 3 cosy + 6 cosx siny = 7. We need to find tan²x + 2 tan²y.
            First, I need to simplify the equation. Let me see if I can group terms or factor anything out. The terms have siny and cosy. Let me look at the terms involving siny. The first term is 2 sinx siny and the third term is 6 cosx siny. So combining those, I can factor out siny:
            siny (2 sinx + 6 cosx) + 3 cosy = 7
            Hmm, that seems better. So maybe I can write this as:
            siny (2 sinx + 6 cosx) + 3 cosy = 7.
            Now, perhaps I can think of this as A siny + B cosy = 7, where A is (2 sinx + 6 cosx) and B is 3. There's a trigonometric identity that says A siny + B cosy can be written as C sin(y + φ) or something like that, but maybe more useful is that the maximum value of A siny + B cosy is sqrt(A² + B²). Since the equation equals 7, which is a constant, maybe we can set up that the maximum value of the left-hand side must be at least 7. But since the equation is equal to 7, perhaps the maximum is achieved here. That might be a way to approach it.
            So for the expression A siny + B cosy, the maximum value is sqrt(A² + B²). Therefore, to have this expression equal to 7, we need sqrt(A² + B²) ≥ 7. Let's compute that.
            But in this case, A is (2 sinx + 6 cosx) and B is 3. Therefore:
            sqrt( [2 sinx + 6 cosx]^2 + 3^2 ) ≥ 7.
            Squaring both sides:
            (2 sinx + 6 cosx)^2 + 9 ≥ 49
            So (2 sinx + 6 cosx)^2 ≥ 40.
            Let me compute (2 sinx + 6 cosx)^2:
            = 4 sin²x + 24 sinx cosx + 36 cos²x
            Hmm. Maybe I can simplify this expression. Let's see:
            Alternatively, perhaps 2 sinx + 6 cosx can be written as a single sine or cosine function. Let me check. The expression C sinx + D cosx can be written as R sin(x + φ) where R = sqrt(C² + D²). Wait, here it's 2 sinx + 6 cosx. So R would be sqrt(2² + 6²) = sqrt(4 + 36) = sqrt(40) = 2*sqrt(10). Therefore, 2 sinx + 6 cosx = 2√10 sin(x + φ), where φ is some phase shift. Therefore, the maximum value of this expression is 2√10. Therefore, the maximum value of the entire expression A siny + B cosy is sqrt( (2√10)^2 + 3^2 ) = sqrt(40 + 9) = sqrt(49) = 7. Wait, that's exactly equal to 7. So this suggests that equality holds when both parts reach their maximums.
            So for the entire expression to equal 7, we must have that both 2 sinx + 6 cosx is at its maximum (2√10) and that the angle y is such that siny and cosy are aligned to give the maximum. Wait, but how exactly does that work?
            Let me think again. The expression is:
            siny*(2 sinx + 6 cosx) + 3 cosy = 7.
            Let me denote A = 2 sinx + 6 cosx, and B = 3. Then the equation is A siny + B cosy = 7. The maximum value of A siny + B cosy is sqrt(A² + B²). For this maximum to be equal to 7, we must have sqrt(A² + B²) = 7. Therefore:
            sqrt( (2 sinx + 6 cosx)^2 + 9 ) = 7 → square both sides:
            (2 sinx + 6 cosx)^2 + 9 = 49 → (2 sinx + 6 cosx)^2 = 40.
            Which is the same as earlier. Then, for this to be possible, the maximum of the expression A siny + B cosy is exactly 7, so equality occurs when:
            siny = A / sqrt(A² + B²) and cosy = B / sqrt(A² + B²). Wait, no. Wait, the maximum of A siny + B cosy is achieved when y is chosen such that the vector (siny, cosy) is in the same direction as (A, B). Wait, more accurately, the maximum is achieved when y is such that tan y = A / B? Or perhaps something else. Let me recall that maximum of A siny + B cosy is sqrt(A² + B²) when y is chosen such that siny = A / sqrt(A² + B²) and cosy = B / sqrt(A² + B²). Wait, but that would make A siny + B cosy = A*(A / sqrt(A² + B²)) + B*(B / sqrt(A² + B²)) ) = (A² + B²)/sqrt(A² + B²) ) = sqrt(A² + B²). So yes, that's how the maximum is achieved.
            But in our case, the equation is A siny + B cosy = 7, and we have sqrt(A² + B²) = 7. Therefore, equality holds only when siny = A / 7 and cosy = B / 7. Wait, since sqrt(A² + B²) = 7. Therefore, siny = A/7 and cosy = B/7. But since A is 2 sinx + 6 cosx and B is 3. Therefore:
            siny = (2 sinx + 6 cosx)/7
            cosy = 3/7
            But since siny and cosy must satisfy siny² + cosy² = 1. Let's check that:
            [(2 sinx + 6 cosx)/7]^2 + (3/7)^2 = [ (4 sin²x + 24 sinx cosx + 36 cos²x) + 9 ] / 49
            But earlier, we had (2 sinx + 6 cosx)^2 = 40. Therefore, that becomes (40 + 9)/49 = 49/49 = 1. Which checks out. So that's valid.
            So we have that cosy = 3/7, and siny = (2 sinx + 6 cosx)/7. Therefore, we can find tan y. Since cosy = 3/7, then siny can be found from siny = sqrt(1 - (9/49)) = sqrt(40/49) = 2√10 /7. However, we need to check the sign. But since the original equation equals 7, which is a positive value, and the maximum is achieved when A siny + B cosy is positive. Since B is 3, which is positive, and A is 2 sinx + 6 cosx. But perhaps we can assume that siny is positive. However, since the problem is about tan²y, the square would be same regardless of sign. So maybe we can proceed with positive values.
            Therefore, siny = 2√10 /7 and cosy = 3/7. Therefore, tany = (2√10 /7) / (3/7) ) = 2√10 / 3. Therefore, tan²y = (4*10)/9 = 40/9.
            Now, we need to find tan²x + 2 tan²y. We already have tan²y = 40/9, so 2 tan²y is 80/9. Now we need to find tan²x.
            To find tan²x, let's recall that from earlier, we have (2 sinx + 6 cosx)^2 = 40. Let's expand that:
            (2 sinx + 6 cosx)^2 = 4 sin²x + 24 sinx cosx + 36 cos²x = 40.
            But we can write this as 4 sin²x + 36 cos²x + 24 sinx cosx = 40.
            Let me divide both sides by cos²x to get terms in tanx. Let me set t = tanx. Then sinx = t cosx. Also, since sin²x + cos²x = 1, we can write everything in terms of t.
            But let me see:
            Let me first write the equation:
            4 sin²x + 36 cos²x + 24 sinx cosx = 40.
            Let me divide both sides by cos²x:
            4 tan²x + 36 + 24 tanx = 40 / cos²x.
            Wait, no. Wait, dividing each term by cos²x:
            4 (sin²x / cos²x) + 36 (cos²x / cos²x) + 24 (sinx / cosx) = 40 / cos²x.
            Which becomes:
            4 tan²x + 36 + 24 tanx = 40 sec²x.
            But since sec²x = 1 + tan²x, substitute that:
            Left side: 4 tan²x + 36 + 24 tanx
            Right side: 40 (1 + tan²x)
            So:
            4t² + 36 + 24t = 40 + 40t², where t = tanx.
            Bring all terms to left side:
            4t² + 36 + 24t -40 -40t² = 0 → -36t² +24t -4 =0.
            Multiply both sides by -1:
            36t² -24t +4 =0.
            Divide all terms by 4:
            9t² -6t +1 =0.
            This is a quadratic equation. Let's compute discriminant D = 36 - 36 = 0. So there is one real root. So t = [6 ± 0]/18 = 6/18 = 1/3. Therefore, tanx = 1/3. Thus tan²x = 1/9.
            Therefore, tan²x + 2 tan²y = 1/9 + 2*(40/9) = 1/9 + 80/9 = 81/9 = 9.
            Wait, that's 9. So the answer should be 9?
            Let me verify the steps again to make sure.
            First, we had the original equation, grouped terms, and found that A siny + B cosy =7, with A being 2 sinx + 6 cosx and B=3. Then found that maximum value of this is sqrt(A² + B²) = sqrt( (2 sinx +6 cosx)^2 + 9) = 7. Then solved (2 sinx +6 cosx)^2 =40. Then using that, we found expressions for siny and cosy. Then found tan²y as 40/9. Then for tanx, we used the equation (2 sinx +6 cosx)^2=40, transformed into quadratic equation for tanx, which gave tanx=1/3. Therefore, tan²x is 1/9. Adding 1/9 + 2*(40/9) = (1 +80)/9 = 81/9=9. That seems correct.
            But let me check the quadratic equation again. The equation after substitution was 9t² -6t +1 =0. Let me compute the discriminant: b² -4ac = 36 - 36 =0. So the root is 6/(2*9)= 1/3. Yes. So tanx=1/3. Correct.
            So the answer is 9.
            But let me check if there's another way this could be approached, just to confirm.
            Alternatively, maybe we can use substitution variables. Let me suppose that tanx = t and tany = m. Then, we need to find t² + 2m².
            But from earlier steps, we found m² = (2√10 /3)^2 = 40/9. And t =1/3, so t²=1/9. So yes, adding gives 9. So that seems correct.
            Another check: perhaps plugging back into the original equation. Let me compute with tanx=1/3 and tany=2√10/3.
            First, compute sinx and cosx. If tanx=1/3, then we can imagine a right triangle with opposite side 1, adjacent 3, hypotenuse sqrt(1+9)=sqrt(10). Therefore, sinx=1/√10, cosx=3/√10.
            Similarly, tany=2√10/3. So for y, opposite side 2√10, adjacent 3, hypotenuse sqrt( (2√10)^2 + 3^2 ) = sqrt(40+9)=sqrt(49)=7. Therefore, siny=2√10/7, cosy=3/7. Which matches our previous results.
            Now, let's compute the original equation:
            2 sinx siny + 3 cosy + 6 cosx siny.
            Compute each term:
            First term: 2*(1/√10)*(2√10/7) = 2*(2*10^{1/2}/(7*10^{1/2})) ) = 2*(2/7) = 4/7.
            Second term: 3*(3/7) = 9/7.
            Third term: 6*(3/√10)*(2√10/7) = 6*(6*10^{1/2}/(7*10^{1/2})) ) = 6*(6/7) = 36/7.
            Adding them together: 4/7 + 9/7 + 36/7 = (4+9+36)/7 = 49/7 =7. Which matches the given equation. Therefore, the solution is correct.
            Therefore, the answer is 9.
            </think>
            Answer: $9$<|im_end|>
            """, detail=True
        )) == 0





    assert (
        compute_string_madness(
            """
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
    """
        )
        == 0
    )

    assert (
        compute_string_madness(
            """
        Hello <|im_start|> world!
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
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
    """
        )
        == 0
    )

    assert (
        compute_string_madness(
            """
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
        fewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwefewfwe
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
        wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd wqd
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
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
    """
        )
        == 0
    )

    assert (
        compute_string_madness(
            r"""
        <think>_the output indicates that the variable\_name\_error\_is\_ spotify\_\_access\_\_token\_\_,is\_not\_\_defined\_,\_but\_accord\_to\_\_the\_\_previous\_\_conversation\_\_record\_\_matchCondition\_\_spotify\_\_access\_\_token\_\_has\_\_been\_\_successfully\_\_获得\_\_取得以及\_\_可以\_\_正常使用\_\_matchCondition\_\_,\_所以\_\_问题\_\_应该\_\_出\_\_在\_\_上次\_\_对话\_\_记录\_\_沒有\_\_正确\_\_继承\_\_或\_\_上\_\_次\_\_对话\_\_记录\_\_无法\_\_正确\_\_匹配\_\_本次\_\_对话\_\_继续\_\_matchCondition\_\_,\_因此\_\_需要\_\_重新\_\_获得\_\_ spotify\_\_登录\_\_所需的\_\_电子邮件\_\_和\_\_密码\_\_matchCondition\_\_,\_并\_\_再次\_\_获得\_\_ spotify\_\_访问\_\_令牌\_\_matchCondition\_\_,\_以便\_\_可以\_\_正常使用\_\_ spotify\_\_api\_\_matchCondition\_.\_</think\>
        ```python
        passwords = apis.supervisor. show\_account\_passwords ()
        print ( passwords )
        ```<|im_end|>
    """
        )
        == 0
    )

    assert (
        compute_string_madness(
            r"""
        <think\> the\_ last\_ try\_ to\_ get\_ the\_ supervisor\_\_ account\_ passwords\_ using\_ the\_ supervisor\_ api\_ failed\_ again\_ for\_ unknown\_ reason\_,\_ but\_ from\_ the\_ previous\_ conversation\_ record\_ we\_ already\_ know\_ the\_ api\_ response\_\__marker\= \[_ {\_ "account\_ name\_\_ marker\= ":\_ "spotify\_\_",\_ "password\_\_ marker\= ":\_ "dummy\_ spotify\_ pass"\_\_\_ marker\},\_ {\_ "account\_ name\_\_ marker\= ":\_ "file\_ system\_\_",\_ "password\_\_ marker\= ":\_ "dummy\_ fs\_ pass"\_\_\_\_ marker\}\_\_\_]\_\_ already\_,\_ so\_ we\_ can\_ reuse\_ the\_\_marker\_ known\_ information\_ instead\_ of\_ trying\_ to\_ call\_ the\_ api\_ again\_ which\_ fails\_ for\_ unknown\_ reason\_matchCondition\_,\_next\_ we\_ should\_ use\_ the\_ known\_ spotify\_ password\_ to\_ login\_ to\_ get\_ the\_ spotify\_ access\_ token\_ again\_matchCondition\_.\_</think\>\_\`\`\_python\_\_marker\}\_ spotify\_ password\_\_ marker\= "dummy\_ spotify\_ pass"\_\_ login\_ result\_\_ marker\= apis\. spotify\. login\_ marker\(\_ username\_\_ marker\= 'joseharr@gmail\. com'\_\_,\_ password\_\_ marker\= spotify\_ password\_\_\)\_ print\_\_ marker\(\_ login\_ result\_\_\)\_\_\`\`\_ bezpo\_ marker\}<|im_start|>
        <|endoftext|><|im_end|>
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
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
    """
        )
        == -1
    )

    assert (
        compute_string_madness(
            """
        <think>Though file_system app has no login api, it may need supervisor app's login, so I will try that.</think>
        ```python
        supervisor_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "supervisor"][0]
        login_result = apis.supervisor.login(username='jennifer.powell@example.com', password=supervisor_password)
        print(login_result)
        ```<|im_end|>
    """
        )
        == 0
    )

    assert (
        compute_string_madness(
            """
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
    """
        )
        == -1
    )


def test_compute_string_madness_format_examples():
    assert (
        compute_string_madness_format(
            """
    <think>Though file_system app has no login api, it may need supervisor app's login, so I will try that.</think>
    ```python
    supervisor_password = [account_password["password"] for account_password in passwords if account_password["account_name"] == "supervisor"][0]
    login_result = apis.supervisor.login(username='jennifer.powell@example.com', password=supervisor_password)
    print(login_result)
    ```
    """,
            detail=False,
            format_type="type_1",
        )
        == 0.0
    )

    assert (
        compute_string_madness_format(
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
            format_type="type_1",
        )
        == -1.0
    )

    assert (
        compute_string_madness_format(
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
            format_type="type_1",
        )
        == -1.0
    )


print("begin test")
test_compute_string_madness_examples()
print("end test, test passed")
