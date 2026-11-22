"""Fixed language-only measurement instructions; no target object labels."""

COMMON = '''Interpret only the supplied referring description. Do not inspect or infer any image, candidate, annotation, dataset identity, or whether a target exists. Extract fallible linguistic structure, not facts about objects.
Return one JSON object, no explanation or Markdown, using lowercase enum values:
{"subject":"person|car|vehicle|other|unknown","subject_span":"exact substring","logic":"and|opaque","terms":[{"kind":"entity|appearance|spatial|motion|role|activity|opaque","name":"entity|color|color_pattern|garment|gender|number|side|team_side|depth|relative_motion|motion|role|activity|pose|other","value":"short canonical value","part":"upper|lower|whole|none|unspecified","binding":"subject|reference|unspecified","reference":"none|image|ego|team|world|other|unspecified","time":"appearance|instant|interval|role_or_action|unspecified","negated":false,"span":"exact contiguous substring"}],"uncertain":["unresolved linguistic issues"]}
Use an entity term for the subject. Keep every meaningful qualifier; use opaque terms rather than deleting unsupported wording. Flat conjunctions may use and. Unresolved disjunctions, quantifiers, or unclear clause structure use opaque and preserve the entire description. All spans must occur literally in the input; canonical values may normalize synonyms (autos to car, grey to gray).
Bind colors to the noun they modify. A red reference car must not become a red subject. Upper/lower garment colors can be separate terms. A multicolor garment such as blue-and-white shorts is one color_pattern, not two mutually exclusive colors. Do not decide that white person means white clothes; mark ambiguous color scope.
Separate position from motion. The left-side team is a team reference, not necessarily an individual on the image left. Same/opposite direction needs a reference; do not insert ego when unstated. Image-side interpretation of bare left/right is an assumption and must be noted in uncertain with reference unspecified. Left in were left can be a verb; preserve that ambiguity. Do not turn pixel displacement into world motion.
Dancing person alone does not specify whether dancing is a role or an instant action: preserve an activity term with time role_or_action and state the ambiguity. Do not silently erase dancing. Garment/color descriptors can have time appearance; this is a description category, not proof a provided TrackID stays the same person. Keep unclear temporal scope unspecified. An unknown clause is not a false clause.
Do not output probabilities, target identities, ground-truth membership or adaptation recommendations. A syntactically valid structure is not claimed correct. Use opaque if unsure. Keep the object concise, at most twelve terms.
'''

PROMPTS = {
    'PARSE': COMMON+'Read the description and extract its subject, bound predicates, references and temporal scopes. Return the JSON object.',
    'SCOPE_REVIEW': COMMON+'Independently check which noun owns each modifier and what each relation refers to. Preserve uncertainty before returning the same JSON schema.'
}
