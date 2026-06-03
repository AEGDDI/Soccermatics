# Do Football Players Pass to Their Own Kind? What Research Says About Cultural Homophily on the Pitch

*How shared culture shapes passing networks — and what Chelsea 2017-18 reveals when you run the numbers yourself*

---

There is a well-known concept in sociology called **homophily**: the tendency of individuals to associate preferentially with others who are similar to themselves. "Birds of a feather flock together," as the saying goes. Sociologists have documented homophily in friendships, marriages, professional networks, and workplaces for decades.

But what about a football pitch? When a midfielder scans his options and decides who to pass to, does the nationality of his teammates unconsciously influence the decision?

On the surface, it seems like it shouldn't. Professional footballers are elite workers with a single, clear objective — win the game. Their incentives are perfectly aligned. Their movements are governed by a tactical system set by the coach. A pass is a calculated action, not a social one.

Yet a recent paper published in *Management Science* — one of the most prestigious journals in economics and management — argues convincingly that cultural homophily is not only present in top-level football, but **persistent, pervasive, and consequential**, even in exactly these "superstar teams" where we'd expect it to disappear.

Here is what the research found, how it was measured, and what happens when you apply the same lens to a specific team.

---

## The Academic Study: Békes & Ottaviano (2025)

Gábor Békés and Gianmarco I. P. Ottaviano assembled a dataset of remarkable scale: **10.73 million passes** made by professional players in the top five European men's leagues — the Premier League, Ligue 1, Bundesliga, Serie A, and La Liga — across eight seasons (2012–13 to 2019–20). That covers 14,608 games and 6,998 players from 138 countries.

Their core question: do players of the same culture pass more to each other than to players of different culture, after controlling for everything else that would naturally explain who passes to whom?

### Defining Culture

The researchers defined culture broadly, not just as nationality. Two players belong to the same cultural group if they share at least one of four traits:

- **Nationality** (same country of citizenship)
- **Colonial legacy** (e.g., Spain and Argentina, or France and Senegal)
- **Federal legacy** (e.g., England, Scotland, and Wales; former Yugoslavia countries)
- **Common language** (including linguistically similar languages)

This broader definition matters because it captures real cultural proximity. A Spanish player and an Argentine player share a language, historical ties, and many cultural references even though they carry different passports. Treating them as culturally distant would miss the point.

By this definition, 50.6% of all player pairs in their dataset share the same culture.

### The Core Finding: A 2.42% Homophily Premium

The naive comparison is simple: same-culture pairs exchange passes at a rate 6.19% higher than different-culture pairs. But this raw number mixes two things: players may pass more to their compatriots simply because coaches tend to field same-culture players together ("induced homophily"), or because players genuinely choose to pass to cultural peers when given the option ("choice homophily").

Békés and Ottaviano go to considerable lengths to isolate choice homophily. Using player-by-half-season fixed effects — effectively controlling for each player's individual passing style, opportunity set, and the composition of teammates they face — they identify a **2.42% homophily premium** that is purely a matter of player choice.

To put this in concrete terms: conditioning on all observable pass and player characteristics, a player is expected to pass 2.42% more to a same-culture teammate than to a different-culture one. And because the homophily premium can be compared to the effect of player market value on passing, the authors calculate that passing to a same-culture receiver is as attractive to the passer as passing to a different-culture receiver who is worth **10.5% more** on the transfer market. That works out to roughly €370,000 for the median player and over €800,000 for the average player.

Homophily has a measurable monetary equivalent. It is not a small effect.

### When Is Homophily Stronger?

The paper doesn't stop at establishing that homophily exists. It asks when it is more or less pronounced — and the answers are revealing.

**Stakes matter.** The homophily premium is significantly larger for long, riskier passes (3.29% vs 1.80% for short passes) and for complex pass sequences — exchanges where the ball goes back and forth between two players as part of a forward move. For these coordinated, high-pressure situations the premium more than doubles: 5.65% versus 2.25% for simple sequences. The more cognitive and physical coordination a pass requires, the more players rely on cultural familiarity.

**Group size matters.** Players who belong to larger cultural groups on the pitch (four or more same-culture teammates) show *more* homophily than players in smaller groups (1.74% vs 3.25%). This goes against the favoritism hypothesis — if homophily were driven by a desire to help your compatriots, you'd expect it to be strongest when there are fewer of them. The opposite pattern supports the cost-saving explanation: in large groups, keeping the ball within the cultural cluster genuinely reduces passing friction.

**Shared experience amplifies it.** This is perhaps the most counterintuitive finding. The contact hypothesis in psychology predicts that repeated interaction between different groups reduces bias. Békés and Ottaviano find the opposite: players who have spent more time together on the same team show *stronger* homophily, not weaker. The longer two same-culture players share a team, the more they pass to each other relative to out-group players. This suggests that familiarity — built through training, socialising, shared routines off the pitch — deepens cultural collaboration rather than diluting it.

### Cost Saving or Favoritism?

The paper engages carefully with a fundamental question: is homophily efficient or not? There are two competing explanations for why players of the same culture might pass more to each other.

The **cost-saving** explanation says passing to a cultural peer is objectively easier because coordination is smoother — shared language, shared football culture, shared intuitions about movement and timing reduce the mental effort of the pass. If this is the mechanism, homophily may actually help the passer help the team.

The **favoritism** explanation says the passer subjectively prefers to keep the ball within his group, regardless of whether it benefits the team. This would be a form of in-group bias that hurts collective performance.

The evidence, while not conclusive, points more toward cost saving than favoritism. Homophily is stronger when stakes are high (under pressure, players revert to the most reliable option), it favors larger groups (consistent with minimizing friction, not with minority solidarity), and shared experience amplifies rather than weakens it (friendship built over time, not instinctive prejudice). Yet the authors are careful to note that the two mechanisms are not mutually exclusive, and fully separating them would require a richer model of team performance.

---

## Translating the Research: Chelsea 2017-18

Academic findings of this scale are compelling precisely because they are aggregate. Ten million passes across five leagues make it very hard to dismiss the result as noise. But aggregate findings can feel abstract. What does a 2.42% homophily premium actually look like inside a specific dressing room?

To bring the research to life, I applied a simpler version of the same logic to Chelsea FC in the 2017-18 Premier League season, using the publicly available Wyscout event data. Chelsea under Antonio Conte is an almost perfect test case: one of the most nationally stratified squads in the league, with a clear, large Spanish-speaking group embedded in a culturally mixed squad.

The nationality breakdown:

| Group | Players |
|-------|---------|
| **Spanish (7)** | Azpilicueta, Marcos Alonso, Caballero, Fàbregas, Pedro, Morata, Arrizabalaga |
| **English (5)** | Cahill, Moses, Barkley, Drinkwater, Green |
| **Italian (2)** | Emerson, Zappacosta |
| **Other (10)** | Hazard, Kanté, Willian, Rüdiger, David Luiz, Courtois, Bakayoko, Christensen, Giroud, Kovačić |

I extracted all accurate passes from Chelsea matches across the season, identified the recipient of each pass using the consecutive-event method, and computed the **Coleman homophily index** for each player:

$$h_g = \frac{w_g - p_g}{1 - p_g}$$

where *w_g* is the fraction of a player's passes going to in-group teammates and *p_g* is the expected fraction under random mixing (simply the group's share of the squad). A value of +1 means exclusively in-group passing; 0 means no preference; negative values mean the player passes *less* within their group than chance would predict.

### What the Numbers Show

The results are nuanced — and more interesting than a simple "Spanish players pass to Spanish players" story.

**Pedro (h = +0.17) and Álvaro Morata (h = +0.13)** are the two Spanish players who show clear positive homophily. Both are wide forwards or strikers — players who depend on their immediate passing partners for combinations, and who in this Chelsea system most frequently connected with the Spanish fullbacks Azpilicueta and Marcos Alonso. The combination of positional proximity and cultural familiarity creates real in-group preference.

**Fàbregas (h ≈ 0.00)** is the neutral hub. The Spanish midfielder functions as the team's central distributor — he connects to everyone, spreads the ball across all nationality groups, and shows zero net cultural preference. His passing map looks like the team's tactical spine, not a cultural clique.

**Azpilicueta (h = −0.09) and Marcos Alonso (h = −0.07)** are both slightly heterophilic. As fullbacks in a 3-4-3/3-5-2 system, they are positionally required to interact with players across the pitch, including Kanté, Christensen, David Luiz, and Cahill in defence and transition. Tactical positioning overrides cultural preference.

**The English players — Cahill (h = −0.15), Moses (h = −0.12), Drinkwater (h = −0.05)** — all show negative homophily. There is no English-language clique at Chelsea under Conte. The English players distribute their passes across cultural groups roughly as the team's tactical structure dictates, without any tendency to cluster.

**Hazard (h = −0.28) and Willian (h = −0.23)** are the most heterophilic players in the squad. As isolated nationalities — Belgian and Brazilian respectively — each with no cultural peers to form a cluster with, they naturally pass across groups. This is consistent with the Békes & Ottaviano finding that players from smaller cultural groups tend to show less (or negative) homophily.

**The Italian pair (h ≈ −0.11)** — Emerson and Zappacosta — almost never pass to each other. They play on opposite flanks, making direct exchange structurally rare, and their small group size provides no basis for cost-saving homophily.

---

## What This Means for Football Analysis

The Békes & Ottaviano paper, and exercises like the Chelsea case study, open up a genuinely new dimension in football analytics.

**Pass networks already show us structure.** Analysts routinely build pass maps to understand a team's tactical shape, identify key connectors, and measure centralisation. What homophily analysis adds is a *social* layer on top of the tactical one: are the strongest passing relationships in a team's network explained by position and role, or does cultural proximity play an independent role?

**The composition of cultural groups has competitive implications.** A team with a dominant cultural group of five or more same-nationality players may show more efficient in-group passing sequences — especially in high-stakes situations — but also more potential fragmentation when those players are absent or injured. A squad composed entirely of cultural isolates (the "Other" group in our Chelsea case) may have more uniform, less cohesive passing structure.

**Managers are already factoring this in — implicitly.** The Békes & Ottaviano paper finds that part of the total homophily premium (1.38%) is "mediated" by managerial decisions: coaches who observe that same-culture pairs collaborate well tend to field them together more often. Antonio Conte signed Emerson and Zappacosta from Italian football, David Luiz from a Portuguese-speaking background, and built his midfield spine around a Franco-Spanish axis of Kanté and Fàbregas. Whether consciously or not, squad composition shapes cultural dynamics on the pitch.

**The colonial legacy finding has underappreciated reach.** Treating Spain and Argentina as the same cultural group for analytical purposes — because their colonial history creates linguistic and cultural overlap — changes how we read passing networks in clubs like Real Madrid or Barcelona, where Argentine, Brazilian and Spanish players regularly co-exist. The shared heritage between Iberian and Latin American players is a real passing signal, not just a biographical footnote.

---

## Conclusions

Football is often presented as a universal language. The research by Békés and Ottaviano suggests it is more like a language family: players communicate more easily and more fluently with cultural neighbours, even when they all share the same ultimate objective and the same tactical instructions.

The effect is modest in absolute terms — 2.42% more passes to same-culture teammates after controlling for everything else. But modest effects accumulate. Across hundreds of passes per match, across an entire season, the preference for same-culture receivers shapes which combinations players trust in tight moments, which partnerships develop naturally, and which players remain peripheral to a team's core passing network.

The Chelsea case study illustrates that this academic finding translates into real, readable patterns in match data: a Spanish group whose two forwards show measurable in-group preference, a central hub who transcends cultural boundaries, a set of isolated nationality players who are by definition the most heterophilic members of the squad.

For football analysts, this is new territory. Passing networks have long been used to map tactical structure. Layering cultural identity onto those networks opens a second reading — one that tells us something about the social fabric underneath the tactics.

---

*The pass network and homophily analysis for Chelsea 2017-18 were produced using the publicly available Wyscout EPL dataset and Python (mplsoccer, matplotlib). The Coleman homophily index was calculated per player across all 38 league matches. Academic source: Békés, G. & Ottaviano, G. I. P. (2025). Cultural Homophily and Collaboration in Superstar Teams. Management Science, 71(10), 8149–8168.*
