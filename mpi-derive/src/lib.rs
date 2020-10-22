#![recursion_limit = "256"]

type TokenStream1 = proc_macro::TokenStream;
type TokenStream2 = proc_macro2::TokenStream;

use quote::quote;
use syn::{Fields, Type};

#[proc_macro_derive(Equivalence)]
pub fn create_user_datatype(input: TokenStream1) -> TokenStream1 {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse struct");
    let result = match ast.data {
        syn::Data::Enum(_) => panic!("#[derive(Equivalence)] is not compatible with enums"),
        syn::Data::Union(_) => panic!("#[derive(Equivalence)] is not compatible with unions"),
        syn::Data::Struct(ref s) => equivalence_for_struct(&ast, &s.fields),
    };
    result.into()
}

fn offset_of(type_ident: &dyn quote::ToTokens, field_name: &dyn quote::ToTokens) -> TokenStream2 {
    quote!(::memoffset::offset_of!(#type_ident, #field_name))
}

fn equivalence_for_tuple_field(type_tuple: &syn::TypeTuple) -> TokenStream2 {
    let field_blocklengths = type_tuple.elems.iter().map(|_| quote! {1 as ::mpi::Count});
    let blocklengths = quote! {[#(#field_blocklengths),*]};

    let field_displacements = type_tuple.elems.iter().enumerate().map(|(i, _)| {
        let field = syn::Index::from(i);
        quote!(::memoffset::offset_of_tuple!(#type_tuple, #field))
    });
    let displacements = quote! {[#(#field_displacements as ::mpi::Address),*]};

    let field_datatypes = type_tuple
        .elems
        .iter()
        .map(|elem| equivalence_for_type(&elem));
    let datatypes = quote! {[#(::mpi::datatype::UncommittedDatatypeRef::from(#field_datatypes)),*]};

    quote! {
        &::mpi::datatype::UncommittedUserDatatype::structured(
            &#blocklengths,
            &#displacements,
            &#datatypes,
        )
    }
}

fn equivalence_for_array_field(type_array: &syn::TypeArray) -> TokenStream2 {
    let ty = equivalence_for_type(&type_array.elem);
    let len = &type_array.len;
    quote! { &::mpi::datatype::UncommittedUserDatatype::contiguous(#len, &#ty) }
}

fn equivalence_for_type(ty: &syn::Type) -> TokenStream2 {
    match ty {
        Type::Path(ref type_path) => quote!(
                <#type_path as ::mpi::datatype::Equivalence>::equivalent_datatype()),
        Type::Tuple(ref type_tuple) => equivalence_for_tuple_field(&type_tuple),
        Type::Array(ref type_array) => equivalence_for_array_field(&type_array),
        _ => panic!("Unsupported type!"),
    }
}

fn equivalence_for_field(field: &syn::Field) -> TokenStream2 {
    equivalence_for_type(&field.ty)
}

fn equivalence_for_struct(ast: &syn::DeriveInput, fields: &Fields) -> TokenStream2 {
    let ident = &ast.ident;

    let field_blocklengths = fields.iter().map(|_| quote! {1 as ::mpi::Count});
    let blocklengths = quote! {[#(#field_blocklengths),*]};

    let field_displacements: Vec<_> = match fields {
        Fields::Named(ref fields) => fields
            .named
            .iter()
            .map(|field| offset_of(&ident, field.ident.as_ref().unwrap()))
            .collect(),
        Fields::Unnamed(ref fields) => fields
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, _)| offset_of(&ident, &syn::Index::from(i)))
            .collect(),
        Fields::Unit => vec![],
    };

    let displacements = quote! {[#(#field_displacements as ::mpi::Address),*]};

    let field_datatypes = fields.iter().map(equivalence_for_field);
    let datatypes = quote! {[#(::mpi::datatype::UncommittedDatatypeRef::from(#field_datatypes)),*]};

    quote! {
        unsafe impl ::mpi::datatype::Equivalence for #ident {
            type Out = ::mpi::datatype::DatatypeRef<'static>;
            fn equivalent_datatype() -> Self::Out {
                use ::mpi::raw::AsRaw;

                thread_local!(static DATATYPE: ::mpi::datatype::DatatypeRef<'static> = {
                    let datatype = ::mpi::datatype::UserDatatype::structured::<
                        ::mpi::datatype::UncommittedDatatypeRef,
                    >(&#blocklengths, &#displacements, &#datatypes);

                    let datatype_ref =
                        unsafe { ::mpi::datatype::DatatypeRef::from_raw(datatype.as_raw()) };

                    ::std::mem::forget(datatype);

                    datatype_ref
                });

                DATATYPE.with(|datatype| datatype.clone())
            }
        }
    }
}
