#![recursion_limit = "256"]

extern crate proc_macro;
extern crate proc_macro2;

#[macro_use]
extern crate quote;
extern crate syn;

type TokenStream1 = proc_macro::TokenStream;
type TokenStream2 = proc_macro2::TokenStream;

#[proc_macro_derive(Datatype)]
pub fn create_user_datatype(input: TokenStream1) -> TokenStream1 {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse struct");
    let result = match ast.data {
        syn::Data::Enum(_) => panic!("#[derive(Datatype)] is not compatible with enums"),
        syn::Data::Union(_) => panic!("#[derive(Datatype)] is not compatible with unions"),
        syn::Data::Struct(ref s) => new_for_struct(&ast, &s.fields),
    };
    result.into()
}

fn new_for_struct(ast: &syn::DeriveInput, fields: &syn::Fields) -> TokenStream2 {
    let ident = &ast.ident;

    let field_count = fields.iter().count();

    let field_blocklengths = fields.iter().map(|_| quote!{1 as ::mpi::Count});
    let blocklengths = quote!{[#(#field_blocklengths),*]};

    let field_displacements = fields.iter().map(|field| {
        let field_ident = field.ident.as_ref().unwrap();
        quote!(
            {
                let value: #ident = unsafe { ::std::mem::uninitialized() };

                let value_loc = &value as *const _ as ::mpi::Address;

                let offset_loc = &value.#field_ident as *const _ as ::mpi::Address;

                ::std::mem::forget(value);

                offset_loc - value_loc
            }
        )
    });
    let displacements = quote!{[#(#field_displacements),*]};

    let field_datatypes = fields.iter().map(|field| {
        let ty = &field.ty;
        quote!(<#ty as ::mpi::datatype::Equivalence>::equivalent_datatype())
    });
    let datatypes_tuple = quote!{(#(#field_datatypes),*)};

    let field_datatype_ref = (0..field_count).map(|field_i| quote!(datatypes.#field_i.as_raw()));
    let datatype_refs = quote!{[#(#field_datatype_ref),*]};

    let count = quote!{#field_count};

    quote!{
        unsafe impl ::mpi::datatype::Equivalence for #ident {
            type Out = ::mpi::datatype::DatatypeRef<'static>;
            fn equivalent_datatype() -> Self::Out {
                use ::mpi::raw::AsRaw;

                thread_local!(static DATATYPE: ::mpi::datatype::DatatypeRef<'static> = {
                    let datatypes = #datatypes_tuple;

                    let blocklengths = #blocklengths;
                    let displacements = #displacements;
                    let types = #datatype_refs;

                    unsafe {
                        let mut newtype: ::mpi::ffi::MPI_Datatype =
                            unsafe { ::std::mem::uninitialized() };
                        ::mpi::ffi::MPI_Type_create_struct(
                            #count as ::mpi::Count,
                            blocklengths.as_ptr(),
                            displacements.as_ptr(),
                            types.as_ptr(),
                            &mut newtype,
                        );
                        ::mpi::ffi::MPI_Type_commit(&mut newtype);
                        ::mpi::datatype::DatatypeRef::from_raw(newtype)
                    }
                });

                DATATYPE.with(|datatype| datatype.clone())
            }
        }
    }
}
