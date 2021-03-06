""" Bibliography formatting tools """
# Import the citeproc-py classes we'll use below.
from citeproc import CitationStylesStyle, CitationStylesBibliography
from citeproc import Citation, CitationItem
from citeproc import formatter
from citeproc.source.json import CiteProcJSON
from citeproc_styles import get_style_filepath
import re


def _flatten(li):
    """ Flatten nested list and remove None values """
    li = [i for i in li if i is not None]
    return [item for sublist in li for item in sublist]


def _uniqueify(li):
    """ Uniqueify based on id """
    return list({v['id']: v for v in li}.values())


def format_bibliography(json_data):
    """ Format CSL-JSON to HTML APA format """
    json_data = _uniqueify(_flatten(json_data))
    bib_source = CiteProcJSON(json_data)
    style_path = get_style_filepath('apa')
    bib_style = CitationStylesStyle(style_path, validate=False)

    # Create the citeproc-py bibliography, passing it the:
    # * CitationStylesStyle,
    # * BibliographySource (CiteProcJSON in this case), and
    # * a formatter (plain, html, or you can write a custom formatter)

    bibliography = CitationStylesBibliography(
        bib_style, bib_source, formatter.html)

    # Processing citations in a document needs to be done in two passes as for
    # some CSL styles, a citation can depend on the order of citations in the
    # bibliography and thus on citations following the current one.
    # For this reason, we first need to register all citations with the
    # CitationStylesBibliography.

    for c in json_data:
        bibliography.register(Citation([CitationItem(c['id'])]))

    items = []
    for item in bibliography.bibliography():
        items.append(str(item))

    return items


def find_predictor_citation(pred, bib):
    """ Search bibliography JSON for entry that matches extractor +
    predictor_name. If ".*", all names will match """
    if pred.extracted_feature is not None:
        ext_name = pred.extracted_feature.extractor_name
        feat_name = pred.name
        for patt, feats in bib.items():
            if re.match(patt, ext_name):
                for fpatt, csl in feats.items():
                    if re.match(fpatt, feat_name):
                        return csl
    return None
