# Refleksionsnoter: hvorfor Retriever Wizard?

Dette dokument er et **arbejdspapir/refleksionspapir**. Det samler motivation, metode og løbende noter om, *hvorfor* Retriever Wizard blev lavet, og *hvordan* værktøjet indgår i min forskningspraksis.

## Problem: behovet for et samlet workflow
Jeg har haft brug for et værktøj, der lader mig undersøge og sammenligne enkelte værker/anskuelsestavler med hele korpusser (anskuelsestavler, kunstværker eller kombinationer).

Tidligere brugte jeg PixPlot, Collection Space Navigator (CSN) og Orange Data Mining. De er brugbare og visuelt “flotte”, men de egner sig ikke til den type arbejdsgang, hvor **nær- og fjernanalyse** kombineres i samme workflow *og* hvor jeg kan gå frem og tilbage mellem: (1) overblik/projektion, (2) konkrete værker, (3) dokumenterede sammenligninger.

Derfor byggede jeg Retriever Wizard: et værktøj der gør det praktisk at arbejde med **Mixed Viewing**.

## Case (hvad jeg bruger det til)
Min overordnede case er anskuelsestavlesamlingen fra DPU/KB.
Jeg ønsker at finde ud af strukturelle ligheder og forskelligheder på tværs af anskuelsestavlesamlingen i relation til de dansk producerede tavler og en bredere medie-kultur.

I praksis bruger jeg Retriever Wizard sammen med clusterprojektioner i CSN til at lave en fortolkning af samlingen: projektionen giver et overblik, og Retriever Wizard understøtter de konkrete “læsninger” og sammenligninger.

## Mixed Viewing (working concept)
Med Mixed Viewing kombinerer jeg klassisk kunsthistorisk værkanalyse med *distant viewing*-metodologi (Tilton & Taylor; Moretti).

Det vigtige for mig er ikke, at det digitale output “forklarer” værkerne, men at det:
- gør det muligt at arbejde i **skala** (overblik på tværs af store mængder)
- og samtidig skaber en konkret og efterprøvbar vej tilbage til **enkeltværker** og **kvalitativ analyse**

## Praktisk arbejdsgang (pipeline)
1. **Computer vision-behandling**
	- Input: et billedkorpus
	- Proces: korpus bearbejdes af en model (fx CLIP eller SigLIP2)
	- Output: **embeddings** (numeriske repræsentationer af billederne)

2. **Overblik og kvalificering (projektion/klynger)**
	- Proces: embeddings projiceres og gennemgås (fx klynger/”clusters” og naboskaber)
	- Formål: at vurdere modellens output i relation til humanistiske analyser
	- Spørgsmål: hvilke spor af bias ser vi? Har det analytisk betydning? Hvordan “forstår” modellen de værktyper, vi er interesserede i, ud fra afstande og naboskaber?

3. **Næranalyse via sammenligning (nearest neighbors)**
	- Proces: jeg tager udgangspunkt i et enkelt værk og undersøger dets nærmeste naboer (visuel lighed) som basis for videre fortolkning.
	- Pointe: værktøjet hjælper mig med at etablere et systematisk sammenligningsfelt, som jeg derefter læser kunsthistorisk.

4. **Analyse af værker og relationer**
	- Resultat: det bliver muligt at sige noget om værkerne i sig selv og deres relationer i datasættet, med både nær- og fjernblik i samme arbejdsgang.


## Hvorfor mixed methods? (kort)
- Anskuelsestavler er et afgrænset, serielt og didaktisk billedformat (ca. 1860–1960), som gør dem egnede til at afprøve tværfaglige metoder.
- Tavlernes klassifikationslogikker har paralleller til AI’s klassifikation/objektgenkendelse, og kan bruges som historisk spejl (og omvendt).
- Format og motivvalg er ofte relativt “læsbart” (didaktiske hensyn), hvilket kan gøre det mere robust i vision-modeller.
- Digitale metoder kan foreslå uventede forbindelser; kvalitativ analyse afgør hvad de betyder.
- Overblik i rum (mange tavler på vægge/salon-ophæng) kan delvist genskabes digitalt og gøre komparation til en konstant mulighed.

## Black box, skala og validering (kort)
- Embeddings/projektioner er **ikke** forklaringer, men et computationelt blik der kan styre opmærksomhed og komparation.
- Udfordringen er fortolkning og autoritet: hvad “tæller” som lighed, og hvilke bias indlejres i data/model/valg?
- Skala er en metodefordel (systematisk komparation på tværs), men bør ikke blive en ny autoritet.
- Målet med Retriever Wizard er at gøre output **efterprøvbart**: tilbage til enkeltværker, naboer og konkrete sammenligninger.

## PixPlot → Retriever Wizard (meget kort)
Jeg startede med PixPlot/Orange/CSN og endte med at bygge et mere sammenhængende workflow, fordi jeg havde brug for at kunne gå frem og tilbage mellem projektion (overblik) og værknære læsninger (sammenligninger/annotering) — og samtidig kunne dokumentere, hvordan forbindelserne opstod.

## Åbne noter (til senere udbygning)
- Udpak: hvad gør anskuelsestavler særligt egnede som empirisk materiale?
- Udpak: hvordan “mixed viewing” relaterer til Tilton & Taylor / Moretti i din konkrete praksis.
- Udpak: hvordan du metodisk validerer en “maskinel” nabo-relation i kunsthistorisk analyse.
